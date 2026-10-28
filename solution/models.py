import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from threadpoolctl import threadpool_limits
from kairos.neural import DenseClassifier
from kairos.history_pool import HistoryClassifier
from kairos.catboost_portable import PortableCatBoost

def macro(y, predicted) -> float:
    confusion = np.bincount(np.asarray(y, dtype=int) * 3 + predicted, minlength=9).reshape(3, 3)
    denominator = confusion.sum(axis=0) + confusion.sum(axis=1)
    return float(np.divide(2 * confusion.diagonal(), denominator,
                           out=np.zeros(3), where=denominator != 0).mean())

def calibrate(y, probabilities) -> tuple[list[float], float]:
    best = np.ones(3)
    best_score = macro(y, probabilities.argmax(axis=1))
    for radius in (1.2, .3, .08):
        center = np.log(best[[0, 2]])
        for low in center[0] + np.linspace(-radius, radius, 9):
            for high in center[1] + np.linspace(-radius, radius, 9):
                weights = np.exp([low, 0, high])
                value = macro(y, (probabilities * weights).argmax(axis=1))
                if value > best_score + 1e-12:
                    best, best_score = weights, value
    return best.tolist(), best_score

def model_frame(frame, schema, config):
    categorical=schema[config['features']+'_categorical']
    columns=schema[config['features']]+categorical
    if config.get('include_text'):columns=columns+['rich_text']
    if {'label','reactions','log_reactions','resid','regression_target','author','post'}&set(columns):
        raise ValueError('Target or identity entered the feature whitelist')
    result=frame[columns].copy()
    if config.get('include_text'):
        result['rich_text']=result['rich_text'].where(result['rich_text'].str.strip().ne(''),'emptydocument')
    for name in categorical:result[name]=result[name].astype(str)
    return result

def stack_features(probabilities, frame, contextual):
    odds = []
    for p in probabilities:
        clipped = np.clip(p, 1e-5, 1)
        odds.append(np.log(clipped[:, [0, 2]] / clipped[:, [1]]))
    odds = np.concatenate(odds, axis=1)
    if not contextual:
        return odds
    gates = np.column_stack([np.clip(frame.level, 0, 10), np.log1p(frame.hist_n),
                             np.clip(frame.hist_iqr, 0, 10), frame.has_reshared_content])
    interactions = (odds[:, :, None] * gates[:, None, :]).reshape(len(frame), -1)
    return np.column_stack([odds, gates, interactions])

def classifier(protocol):
    return make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler(),
                         LogisticRegression(C=protocol['C'], max_iter=protocol['max_iter'], solver='lbfgs'))

class EmbeddedHistory:
    """Use the original network and preprocessing without global workspace lookups."""

    def __init__(self, model, lookup, arrays):
        self.model, self.lookup, self.arrays = model, lookup, arrays
        self.classes_ = model.classes_
        self._data = None

    @classmethod
    def from_model(cls, model):
        data = model.resources('cpu')
        arrays = {key: value.cpu().numpy().copy() for key, value in data.items() if torch.is_tensor(value)}
        lookup = data['lookup'].copy()
        model.resources_ = None
        return cls(model, lookup, arrays)

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_data'] = None
        return state

    def predict_proba(self, frame):
        if self._data is None:
            self._data = {'lookup': self.lookup, 'device': 'cpu',
                          **{key: torch.from_numpy(value) for key, value in self.arrays.items()}}
        data = self._data
        metadata = torch.from_numpy(np.asarray(self.model.processing_.transform(self.model.metadata(frame)), dtype='float32'))
        rows = self.model.rows(frame, data)
        output = []
        self.model.network_.eval()
        with torch.inference_mode():
            for start in range(0, len(frame), 512):
                logits = self.model.network_(metadata[start:start+512],
                    *self.model.batch(data, rows[start:start+512]))
                output.append(torch.softmax(logits, dim=1).numpy())
        return np.concatenate(output)

class StackBundle:
    def __init__(self, components, meta, variant, weights, biases=None):
        self.components, self.meta, self.variant = components, meta, variant
        self.weights = np.asarray(weights, dtype='float64')
        self.biases = np.asarray(biases if biases is not None else [0., 0., 0.], dtype='float64')
        self.classes_ = np.array([0, 1, 2])

    def predict_proba(self, frames):
        values, reference = [], None
        for component in self.components:
            frame = frames[component['cache']]
            keys = frame[['author', 'post']].reset_index(drop=True)
            if keys.duplicated().any() or not len(keys):
                raise ValueError('Inference keys must be nonempty and unique')
            if reference is None:
                reference, gates = keys, frame
            else:
                pd.testing.assert_frame_equal(reference, keys, check_dtype=False)
            model = component['model']
            p = (model.predict_proba(frame) if isinstance(model, EmbeddedHistory) else
                 probabilities(model, frame, component['schema'], component['config']))
            values.append(checked_probabilities(p, len(frame)))
        return checked_probabilities(self.meta.predict_proba(stack_features(values, gates, self.variant == 'context')), len(gates))

    def predict(self, frames):
        return (self.predict_proba(frames) * self.weights + self.biases).argmax(1)

def checked_probabilities(values, rows):
    values=np.asarray(values)
    if values.shape!=(rows,3) or not np.isfinite(values).all() or (values<0).any():
        raise ValueError("Invalid three-class probabilities")
    if not np.allclose(values.sum(axis=1),1,atol=1e-5):
        raise ValueError("Probabilities do not sum to one")
    return values


def fit_model(config,schema,frame,seed):
    if config['kind']=='history':
        return HistoryClassifier(schema,config,seed).fit(frame,frame.label)
    x=model_frame(frame,schema,config)
    categorical=[name for name in x if not pd.api.types.is_numeric_dtype(x[name]) and name!='rich_text']
    if config['kind']=='mlp':
        numeric=[name for name in x if name not in categorical]
        processing=ColumnTransformer([
            ('num',Pipeline([('fill',SimpleImputer(strategy='median',keep_empty_features=True,add_indicator=True)),('scale',StandardScaler())]),numeric),
            ('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False,dtype=np.float32),categorical)],sparse_threshold=0)
        classifier=DenseClassifier(hidden=config['hidden'],epochs=config['epochs'],rate=config['rate'],
            dropout=config['dropout'],decay=config['decay'],balanced=config['balanced'],seed=seed,
            normalization=config.get('normalization','layer'))
        model=Pipeline([('features',processing),('classifier',classifier)])
        with threadpool_limits(limits=4):model.fit(x,frame.label)
        return model
    if config['kind']!='cat':raise ValueError('Unsupported selected model')
    from catboost import CatBoostClassifier
    model=CatBoostClassifier(iterations=config['iterations'],depth=config['depth'],learning_rate=config['rate'],
        l2_leaf_reg=config['l2'],class_weights=None,loss_function='MultiClass',thread_count=4,
        random_seed=seed,allow_writing_files=False,verbose=False)
    with threadpool_limits(limits=4):
        model.fit(x,frame.label,cat_features=categorical,text_features=['rich_text'])
    return PortableCatBoost.from_fitted(model)


def probabilities(model,frame,schema,config):
    with threadpool_limits(limits=4):
        values=model.predict_proba(frame) if config['kind']=='history' else model.predict_proba(model_frame(frame,schema,config))
    return checked_probabilities(values,len(frame))
