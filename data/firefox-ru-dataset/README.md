# *Русский* &mdash; Russian (`ru`)

This datasheet is for cv-corpus-25.0-2026-03-09 of the Mozilla Common Voice *Scripted Speech* dataset for Russian [Русский - `ru`]. The dataset contains 201947 clips representing 290.51 hours of recorded speech (251.94 hours validated) from 3695 speakers, recorded from a text corpus of 48,092 sentences.

## Language

### Accents

| Code | Accent | Clips | Speakers |
|---|---|---|---|
| - |  | 88,967 (44.1%) | 209 (5.7%) |

## Demographic information

The dataset includes the following self-declared age and gender distributions. A coverage summary is shown below each table.

### Gender

Self-declared gender information. The table shows clip and speaker counts with percentages. Speakers who did not declare a gender are listed as Unspecified. A dash (-) indicates zero.

| Code | Gender | Clips | Speakers |
|---|---|---|---|
| male_masculine | Male, masculine | 119,644 (59.2%) | 1,023 (27.7%) |
| female_feminine | Female, feminine | 29,884 (14.8%) | 358 (9.7%) |
| transgender | Transgender | - | - |
| non-binary | Non-binary | 40 (0.0%) | 2 (0.1%) |
| do_not_wish_to_say | Prefer not to say | - | - |
| - | Unspecified | 52,379 (25.9%) | 2,507 (67.8%) |

*Gender declared: 149,568 of 201,947 clips (74.1%), 1,188 of 3,695 speakers (32.2%)*

### Age

Self-declared age information. The table shows clip and speaker counts with percentages. Speakers who did not declare an age are listed as Unspecified. A dash (-) indicates zero.

| Code | Age | Clips | Speakers |
|---|---|---|---|
| teens | Teens | 18,658 (9.2%) | 188 (5.1%) |
| twenties | Twenties | 65,935 (32.6%) | 690 (18.7%) |
| thirties | Thirties | 36,315 (18.0%) | 412 (11.2%) |
| fourties | Fourties | 27,010 (13.4%) | 136 (3.7%) |
| fifties | Fifties | 5,905 (2.9%) | 30 (0.8%) |
| sixties | Sixties | 231 (0.1%) | 13 (0.4%) |
| seventies | Seventies | 5 (0.0%) | 1 (0.0%) |
| eighties | Eighties | - | - |
| nineties | Nineties | - | - |
| - | Unspecified | 47,888 (23.7%) | 2,440 (66.0%) |

*Age declared: 154,059 of 201,947 clips (76.3%), 1,255 of 3,695 speakers (34.0%)*

## Data splits for modelling

**Clip buckets**

| Bucket | Clips |
|---|---|
| Validated | 175,137 (86.7%) |
| Invalidated | 10,749 (5.3%) |
| Other | 16,061 (8.0%) |

**Training splits**

| Split | Clips |
|---|---|
| Train | 26,920 (15.4%) |
| Dev | 10,282 (5.9%) |
| Test | 10,283 (5.9%) |

*Training split coverage: 47,485 of 175,137 validated clips (27.1%)*

The dataset contains 175137 validated, 10749 invalidated, and 16061 unresolved clips. The average clip duration is 5.179 seconds.

## Text corpus

**Validated sentences:** 47,850

| Category | Count |
|---|---|
| Unvalidated sentences | 242 |
| Pending sentences | 4 |
| Rejected sentences | 238 |
| Reported sentences | 572 |

The corpus contains 48,092 sentences: 47,850 validated and 242 unvalidated (4 pending review, 238 rejected), with 572 reported for review.

### Sample

There follows a randomly selected sample of five sentences from the corpus.

1. *Он был основоположником того, что сегодня мы называем миротворческими силами.*
2. *Многих детей учат ненависти.*
3. *Любой даже самый совершенный и полный словарь всегда отстаёт от живого языка.*
4. *Честное слово, не потом!*
5. *тридцать четыре груши*

### Sources

| Source | Sentences |
|---|---|
| sentence-collector | 46,118 (96.4%) |
| самоцитирование | 812 (1.7%) |
| highsource-de-ru-translations | 499 (1.0%) |
| Other | 421 (0.9%) |

### Text domains

| Code | Domain | Clips | Speakers |
|---|---|---|---|
| general | General | 43 (0.0%) | 28 (0.8%) |
| agriculture_food | Agriculture and Food | 4 (0.0%) | 4 (0.1%) |
| automotive_transport | Automotive and Transport | 4 (0.0%) | 4 (0.1%) |
| finance | Finance | 8 (0.0%) | 8 (0.2%) |
| service_retail | Service and Retail | 12 (0.0%) | 11 (0.3%) |
| healthcare | Healthcare | 7 (0.0%) | 7 (0.2%) |
| history_law_government | History, Law and Government | 48 (0.0%) | 27 (0.7%) |
| media_entertainment | Media and Entertainment | 22 (0.0%) | 18 (0.5%) |
| nature_environment | Nature and Environment | 6 (0.0%) | 6 (0.2%) |
| news_current_affairs | News and Current Affairs | 5 (0.0%) | 5 (0.1%) |
| technology_robotics | Technology and Robotics | 23 (0.0%) | 18 (0.5%) |
| language_fundamentals | Language Fundamentals | 4 (0.0%) | 4 (0.1%) |

### Fields

#### Clips

Each row of a `tsv` file represents a single audio clip, and contains the following information:

- `client_id` - hashed UUID of a given user
- `path` - relative path of the audio file
- `text` - supposed transcription of the audio
- `up_votes` - number of people who said audio matches the text
- `down_votes` - number of people who said audio does not match text
- `age` - age of the speaker[^1]
- `gender` - gender of the speaker[^1]
- `accents` - accents of the speaker[^1]
- `variant` - variant of the language[^1]
- `segment` - if sentence belongs to a custom dataset segment, it will be listed here
- `prompt_upvotes` - number of upvotes the sentence prompt received
- `prompt_reports` - number of reports the sentence prompt received
- `is_edited` - whether the clip's transcription has been edited

[^1]: For a full list of age, gender, and accent options, see the [demographics spec](https://github.com/common-voice/common-voice/blob/main/web/src/stores/demographics.ts). These will only be reported if the speaker opted in to provide that information.

#### `validated_sentences.tsv`

The `validated_sentences.tsv` file contains one row per validated sentence in the text corpus:

- `sentence_id` - unique identifier for the sentence
- `sentence` - the sentence text
- `variant` - the variant of the language
- `sentence_domain` - the domain(s) the sentence belongs to
- `source` - the source the sentence was collected from
- `is_used` - whether the sentence is still in circulation for recording
- `clips_count` - number of clips recorded for this sentence

#### `unvalidated_sentences.tsv`

The `unvalidated_sentences.tsv` file contains one row per unvalidated sentence in the text corpus:

- `sentence_id` - unique identifier for the sentence
- `sentence` - the sentence text
- `variant` - the variant of the language
- `sentence_domain` - the domain(s) the sentence belongs to
- `source` - the source the sentence was collected from
- `up_votes` - number of upvotes the sentence received
- `down_votes` - number of downvotes the sentence received
- `status` - current status of the sentence (`pending` or `rejected`)

## Get involved

### Community links

- [Common Voice translators on Pontoon](https://pontoon.mozilla.org/ru/common-voice/contributors/)
- [Common Voice Communities](https://github.com/common-voice/common-voice/blob/main/docs/COMMUNITIES.md)

### Discussions

- [Common Voice on Matrix](https://chat.mozilla.org/#/room/#common-voice:mozilla.org)
- [Common Voice on Discourse](https://discourse.mozilla.org/t/about-common-voice-readme-first/17218)
- [Common Voice on Discord](https://discord.gg/9QTj9zwn)
- [Common Voice on Telegram](https://t.me/mozilla_common_voice)

### Contribute

- [Speak](https://commonvoice.mozilla.org/ru/speak)
- [Write](https://commonvoice.mozilla.org/ru/write)
- [Listen](https://commonvoice.mozilla.org/ru/listen)
- [Review](https://commonvoice.mozilla.org/ru/review)

## Licence

This dataset is released under the [Creative Commons Zero (CC-0)](https://creativecommons.org/public-domain/cc0/) licence. By downloading this data you agree to not determine the identity of speakers in the dataset.
