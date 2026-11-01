from __future__ import annotations
import re
import unicodedata
from collections import Counter
from urllib.parse import urlsplit
import emoji
import numpy as np
import pandas as pd
FEATURE_NAMES = ['text_n_chars', 'text_n_words', 'text_n_lines', 'text_n_sentences', 'text_mean_word_chars', 'text_unique_word_ratio', 'text_upper_ratio', 'text_upper_word_count', 'text_digit_ratio', 'text_punctuation_ratio', 'text_exclamation_count', 'text_question_count', 'text_ellipsis_count', 'text_repeated_punctuation_count', 'text_elongated_token_count', 'text_max_repeat_run', 'text_markdown_emphasis_count', 'text_hashtag_count', 'text_hashtag_unique_count', 'text_hashtag_mean_chars', 'text_mention_count', 'text_url_count', 'text_email_count', 'text_placeholder_count', 'text_url_youtube_count', 'text_url_x_count', 'text_url_linkedin_count', 'text_url_other_count', 'text_emoji_count', 'text_emoji_unique_count', 'text_emoji_type_count', 'text_digit_token_count', 'text_date_mention_count', 'text_time_mention_count', 'text_money_mention_count', 'text_money_amount_log_max', 'text_money_currency_count', 'text_is_empty', 'text_word_length_std', 'text_paragraph_count', 'text_first_line_chars', 'text_bullet_line_count', 'text_numbered_line_count', 'text_quote_line_count', 'text_latin_ratio', 'text_cyrillic_ratio', 'text_cjk_ratio', 'text_unicode_changed_ratio', 'text_emoji_per_word', 'text_long_word_ratio', 'text_negation_count', 'text_intensifier_count', 'text_question_word_count', 'text_first_person_count', 'text_second_person_count', 'text_cta_count', 'text_smile_emoticon_count', 'text_frown_emoticon_count', 'text_laugh_emoticon_count', 'text_bow_emoticon_count', 'text_happy_face_count']
_EMAIL_RE = re.compile('(?i)\\b[\\w.+%-]+@[\\w.-]+\\.[^\\W\\d_]{2,}\\b', re.UNICODE)
_URL_RE = re.compile('(?i)(?:https?://|www\\.)[^\\s<>]+|(?<![\\w@])(?:[\\w](?:[\\w-]{0,61}[\\w])?\\.)+[^\\W\\d_](?:[^\\W_]|-){1,62}(?::\\d{1,5})?(?:/[^\\s<>]*)?')
_MENTION_RE = re.compile('(?<![\\w.+%-])@[\\w.-]{1,64}', re.UNICODE)
_HASHTAG_RE = re.compile('(?<!\\w)#([\\w\\u200c\\u200d]+)', re.UNICODE)
_WORD_RE = re.compile("[^\\W_]+(?:['’][^\\W_]+)?", re.UNICODE)
_PLACEHOLDER_RE = re.compile('[⟨<](?:URL|HANDLE|EMAIL|PHONE|ACCOUNT|PERSON(?:_[^⟩>]+)?|AUTHOR(?:_[^⟩>]+)?|ORG(?:_[^⟩>]+)?)[⟩>]', re.I)
_DATE_RE = re.compile('(?i)\\b(?:\\d{1,4}[./-]\\d{1,2}[./-]\\d{1,4}|(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\\s+\\d{1,2}(?:st|nd|rd|th)?[,]?\\s+\\d{4})\\b')
_TIME_RE = re.compile('(?i)\\b(?:[01]?\\d|2[0-3]):[0-5]\\d\\s*(?:am|pm)?\\b')
_MONEY_RE = re.compile('(?P<currency>[$€£₽¥])\\s*(?P<amount>\\d[\\d, .]*\\d|\\d)')
_REPEAT_RE = re.compile('(.)\\1{2,}', re.UNICODE)
_ELONGATED_TOKEN_RE = re.compile('(?iu)\\b[^\\W_]*([^\\W_])\\1{2,}[^\\W_]*\\b', re.UNICODE)
_PUNCT_RUN_RE = re.compile('([!?.,])\\1{1,}')
_EMPHASIS_RE = re.compile('(?<!\\w)(\\*{1,2}|_{1,2})(?=\\S)(.+?)(?<=\\S)\\1(?!\\w)')
_HASHTAG_SPLIT_1 = re.compile('([a-z])([A-Z])')
_HASHTAG_SPLIT_2 = re.compile('([A-Z]+)([A-Z][a-z])')
_ZERO_WIDTH_RE = re.compile('[\u200b\u200e\u200f\ufeff]')
_CURRENCY_NAMES = {'$': 'DOLLAR', '€': 'EURO', '£': 'POUND', '₽': 'RUBLE', '¥': 'YEN_YUAN'}
for _currency_name in _CURRENCY_NAMES.values():
    FEATURE_NAMES.extend((f'text_money_{_currency_name.lower()}_count', f'text_money_{_currency_name.lower()}_log_max'))
_EMOTICONS = ((re.compile('(?i)(?<!\\w)(?::-?\\)|;[-)]|\\(:)(?!\\w)'), ' smile '), (re.compile('(?i)(?<!\\w)(?::-?\\(|;[-(])(?!\\w)'), ' frown '), (re.compile('(?i)(?<!\\w)(?:x-?d|:d)(?!\\w)'), ' laugh '), (re.compile('m\\s*\\(\\s*_\\s*_\\s*\\)\\s*m', re.I), ' bow '), (re.compile('٩\\s*\\(◕‿◕｡\\)۶'), ' happy_face '))
_LEXICONS = {'negation': frozenset('no not never neither nor cannot не нет никогда нельзя ни без'.split()), 'intensifier': frozenset('very really extremely absolutely totally so очень крайне абсолютно совсем действительно'.split()), 'question_word': frozenset('who what why when where how which кто что почему когда где как какой'.split()), 'first_person': frozenset('i me my mine we us our ours я мы меня мой наш нам мне'.split()), 'second_person': frozenset('you your yours ты вы тебя вас твой ваш вам тебе'.split()), 'cta': frozenset('subscribe follow comment share click join register download sign подпишитесь подписывайтесь поделитесь напишите скачайте присоединяйтесь зарегистрируйтесь'.split())}

def _nfkc(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ''
    return _ZERO_WIDTH_RE.sub('', unicodedata.normalize('NFKC', str(value)))

def _hashtag_parts(value: str) -> list[str]:
    spaced = value.replace('_', ' ')
    spaced = _HASHTAG_SPLIT_2.sub('\\1 \\2', spaced)
    spaced = _HASHTAG_SPLIT_1.sub('\\1 \\2', spaced)
    return [part for part in _WORD_RE.findall(spaced) if part]

def _safe_surface(value: str) -> tuple[str, list[str], list[str]]:
    safe = _EMAIL_RE.sub(' <EMAIL> ', value)
    safe = _URL_RE.sub(' <URL> ', safe)
    safe = _MENTION_RE.sub(' <USER> ', safe)
    safe = re.sub('[⟨<](?:PERSON|AUTHOR|ORG)(?:_[^⟩>]*)?[⟩>]', ' <ENTITY> ', safe, flags=re.I)
    safe = re.sub('[⟨<]HANDLE[⟩>]', ' <USER> ', safe, flags=re.I)
    safe = re.sub('[⟨<]PHONE(?:_[^⟩>]*)?[⟩>]', ' <PHONE> ', safe, flags=re.I)
    safe = re.sub('[⟨<]ACCOUNT(?:_[^⟩>]*)?[⟩>]', ' <ACCOUNT> ', safe, flags=re.I)
    safe = re.sub('[⟨<]URL[⟩>]', ' <URL> ', safe, flags=re.I)
    safe = re.sub('[⟨<]EMAIL[⟩>]', ' <EMAIL> ', safe, flags=re.I)
    hashtags = [m.group(1) for m in _HASHTAG_RE.finditer(safe)]
    emoji_values = [item['emoji'] for item in emoji.emoji_list(safe)]
    return (safe, hashtags, emoji_values)

def _url_platform_counts(source: str) -> tuple[int, int, int, int]:
    youtube = x_platform = linkedin = other = 0
    for match in _URL_RE.finditer(source):
        candidate = match.group(0).rstrip('.,;:!?)]}>')
        try:
            parsed = urlsplit(candidate if '://' in candidate else f'//{candidate}')
            host = (parsed.hostname or '').casefold().removeprefix('www.')
        except ValueError:
            other += 1
            continue
        if host in {'youtube.com', 'youtu.be', 'm.youtube.com'}:
            youtube += 1
        elif host in {'x.com', 'twitter.com', 't.co'}:
            x_platform += 1
        elif host == 'linkedin.com' or host.endswith('.linkedin.com'):
            linkedin += 1
        else:
            other += 1
    return (youtube, x_platform, linkedin, other)

def _normalized_text(safe_surface: str) -> str:
    text = safe_surface
    text = _DATE_RE.sub(' <DATE> ', text)
    text = _TIME_RE.sub(' <TIME> ', text)

    def normalize_money(match: re.Match[str]) -> str:
        code = _CURRENCY_NAMES.get(match.group('currency'), 'CURRENCY')
        return f' <CURRENCY_{code}> <AMOUNT> '
    text = _MONEY_RE.sub(normalize_money, text)
    for pattern, replacement in _EMOTICONS:
        text = pattern.sub(replacement, text)
    text = _REPEAT_RE.sub(lambda m: m.group(1) * 2, text)
    text = _PUNCT_RUN_RE.sub(lambda m: m.group(1) * 2, text)

    def segment(match: re.Match[str]) -> str:
        tag = match.group(1)
        parts = _hashtag_parts(tag)
        return f"#{tag} {' '.join(parts)}" if parts else f'#{tag}'
    text = _HASHTAG_RE.sub(segment, text)
    try:
        text = emoji.demojize(text, language='en')
    except (KeyError, ValueError):
        pass
    return text

def _money_values(text: str) -> tuple[int, float, int]:
    amounts, currencies = ([], set())
    matches = list(_MONEY_RE.finditer(text))
    for match in matches:
        currency = match.group('currency')
        currencies.add(currency)
        amount = re.sub('\\s', '', match.group('amount'))
        if ',' in amount and '.' in amount:
            if amount.rfind(',') > amount.rfind('.'):
                amount = amount.replace('.', '').replace(',', '.')
            else:
                amount = amount.replace(',', '')
        elif ',' in amount:
            tail = amount.rsplit(',', 1)[-1]
            amount = amount.replace(',', '' if len(tail) == 3 else '.')
        elif amount.count('.') > 1:
            pieces = amount.split('.')
            amount = ''.join(pieces[:-1]) + ('.' + pieces[-1] if len(pieces[-1]) != 3 else pieces[-1])
        try:
            amounts.append(float(amount))
        except ValueError:
            continue
    return (len(matches), float(np.log1p(max(amounts, default=0.0))), len(currencies))

def _surface_features(source: object, safe: str, hashtags: list[str], emojis: list[str]) -> dict[str, float]:
    source = '' if source is None or pd.isna(source) else str(source)
    source_nfkc = _nfkc(source)
    prose = re.sub('[⟨<][^⟩>]+[⟩>]', ' ', safe)
    words = _WORD_RE.findall(prose)
    chars = list(safe)
    non_space = [ch for ch in chars if not ch.isspace()]
    punctuation = [ch for ch in chars if unicodedata.category(ch).startswith('P')]
    repeated_runs = list(_REPEAT_RE.finditer(safe))
    emoji_counts = Counter(emojis)
    money_count, money_log_sum, currency_count = _money_values(safe)
    date_count = len(_DATE_RE.findall(source_nfkc))
    time_count = len(_TIME_RE.findall(source_nfkc))
    hashtag_lens = [len(tag) for tag in hashtags]
    upper_words = sum((1 for word in words if len(word) > 1 and word.isupper()))
    placeholder_count = len(_PLACEHOLDER_RE.findall(safe)) + safe.count('<ENTITY>') + safe.count('<USER>')
    source_n_mentions = len(_MENTION_RE.findall(source_nfkc)) + len(re.findall('[⟨<](?:HANDLE|PERSON|AUTHOR)[^⟩>]*[⟩>]', source_nfkc, re.I))
    source_n_urls = len(_URL_RE.findall(source_nfkc)) + len(re.findall('[⟨<]URL[⟩>]', source_nfkc, re.I))
    url_platforms = _url_platform_counts(_EMAIL_RE.sub(' ', source_nfkc))
    source_n_emails = len(_EMAIL_RE.findall(source_nfkc)) + len(re.findall('[⟨<]EMAIL[⟩>]', source_nfkc, re.I))
    alpha_count = max(1, sum((ch.isalpha() for ch in prose)))
    word_counts = Counter((word.casefold() for word in words))
    lines = safe.splitlines()
    values: dict[str, float] = {'text_n_chars': float(len(safe)), 'text_n_words': float(len(words)), 'text_n_lines': float(safe.count('\n') + 1 if safe else 0), 'text_n_sentences': float(len(re.findall('[.!?]+(?=\\s|$)', safe))), 'text_mean_word_chars': float(np.mean([len(w) for w in words])) if words else 0.0, 'text_unique_word_ratio': float(len({w.casefold() for w in words}) / len(words)) if words else 0.0, 'text_upper_ratio': float(sum((ch.isupper() for ch in prose)) / alpha_count), 'text_upper_word_count': float(upper_words), 'text_digit_ratio': float(sum((ch.isdigit() for ch in safe)) / max(1, len(non_space))), 'text_punctuation_ratio': float(len(punctuation) / max(1, len(non_space))), 'text_exclamation_count': float(safe.count('!')), 'text_question_count': float(safe.count('?')), 'text_ellipsis_count': float(safe.count('…') + len(re.findall('\\.{3,}', safe))), 'text_repeated_punctuation_count': float(sum((len(m.group(0)) - 1 for m in _PUNCT_RUN_RE.finditer(safe)))), 'text_elongated_token_count': float(len(_ELONGATED_TOKEN_RE.findall(safe))), 'text_max_repeat_run': float(max((len(m.group(0)) for m in repeated_runs), default=int(bool(safe)))), 'text_markdown_emphasis_count': float(len(_EMPHASIS_RE.findall(safe))), 'text_hashtag_count': float(len(hashtags)), 'text_hashtag_unique_count': float(len({h.casefold() for h in hashtags})), 'text_hashtag_mean_chars': float(np.mean(hashtag_lens)) if hashtag_lens else 0.0, 'text_mention_count': float(source_n_mentions), 'text_url_count': float(source_n_urls), 'text_email_count': float(source_n_emails), 'text_placeholder_count': float(placeholder_count), 'text_url_youtube_count': float(url_platforms[0]), 'text_url_x_count': float(url_platforms[1]), 'text_url_linkedin_count': float(url_platforms[2]), 'text_url_other_count': float(url_platforms[3]), 'text_emoji_count': float(len(emojis)), 'text_emoji_unique_count': float(len(emoji_counts)), 'text_emoji_type_count': float(len({emoji.demojize(e, language='en').strip(':').split('_')[0] for e in emojis})), 'text_digit_token_count': float(sum((any((ch.isdigit() for ch in word)) for word in words))), 'text_date_mention_count': float(date_count), 'text_time_mention_count': float(time_count), 'text_money_mention_count': float(money_count), 'text_money_amount_log_max': float(money_log_sum), 'text_money_currency_count': float(currency_count), 'text_is_empty': float(not bool(safe.strip())), 'text_word_length_std': float(np.std([len(w) for w in words])) if words else 0.0, 'text_paragraph_count': float(len(re.split('\\n\\s*\\n', safe.strip()))) if safe.strip() else 0.0, 'text_first_line_chars': float(len(lines[0])) if lines else 0.0, 'text_bullet_line_count': float(sum((bool(re.match('\\s*[-*•▪→] ', line)) for line in lines))), 'text_numbered_line_count': float(sum((bool(re.match('\\s*\\d+[.)]\\s', line)) for line in lines))), 'text_quote_line_count': float(sum((line.lstrip().startswith('>') for line in lines))), 'text_latin_ratio': float(len(re.findall('[a-zA-Z]', prose)) / alpha_count), 'text_cyrillic_ratio': float(len(re.findall('[\\u0400-\\u052f]', prose)) / alpha_count), 'text_cjk_ratio': float(len(re.findall('[\\u3400-\\u9fff\\u3040-\\u30ff\\uac00-\\ud7af]', prose)) / alpha_count), 'text_unicode_changed_ratio': float(sum((unicodedata.normalize('NFKC', ch) != ch for ch in source)) / max(1, len(source))), 'text_emoji_per_word': float(len(emojis) / max(1, len(words))), 'text_long_word_ratio': float(sum((len(w) >= 12 for w in words)) / max(1, len(words)))}
    for name, lexicon in _LEXICONS.items():
        values[f'text_{name}_count'] = float(sum((word_counts[word] for word in lexicon)))
    for currency, name in _CURRENCY_NAMES.items():
        currency_text = ' '.join((match.group(0) for match in _MONEY_RE.finditer(safe) if match.group('currency') == currency))
        count, maximum, _ = _money_values(currency_text)
        values[f'text_money_{name.lower()}_count'] = float(count)
        values[f'text_money_{name.lower()}_log_max'] = maximum
    for name, (pattern, _) in zip(('smile_emoticon', 'frown_emoticon', 'laugh_emoticon', 'bow_emoticon', 'happy_face'), _EMOTICONS):
        values[f'text_{name}_count'] = float(len(pattern.findall(safe)))
    return values

def prepare_texts(texts: pd.Series) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    surface_rows: list[str] = []
    normalized_rows: list[str] = []
    feature_rows: list[dict[str, float]] = []
    for value in texts.fillna(''):
        source = _nfkc(value)
        safe, hashtags, emoji_values = _safe_surface(source)
        surface_rows.append(safe)
        normalized_rows.append(_normalized_text(safe))
        feature_rows.append(_surface_features(value, safe, hashtags, emoji_values))
    idx = texts.index
    dense = pd.DataFrame(feature_rows, index=idx, columns=FEATURE_NAMES).astype('float32')
    dense = dense.replace([np.inf, -np.inf], 0).fillna(0)
    return (pd.Series(surface_rows, index=idx, name='safe_surface_text', dtype='string'), pd.Series(normalized_rows, index=idx, name='normalized_text', dtype='string'), dense)
