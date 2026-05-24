# NewsDigest: News Collection and Automatic Summarisation

## NLP task and domain/application area
Task: text_summarization. Domain: digital news media. Application area: automatic single-document summarisation of English news articles.

## Motivation and problem statement
Digital news volume makes manual scanning infeasible; readers spend significant time finding the most relevant information. Given a single English news article, produce a short summary that is concise, readable, and faithful to the source.

## Expected final product
A prototype that accepts an article or news URL and returns a generated summary, with a comparison of methods and an evaluation of quality. It improves: It reduces the time required to read full-length articles.

## Research questions
- RQ1: How effectively can an NLP system summarise newspaper articles concisely and informatively?
- RQ2: Does BART outperform extractive baselines (Lead-3, TextRank) in ROUGE on CNN/DailyMail and XSum?
- RQ3: Which article characteristics drive summary-quality differences between methods?

## Dataset
Source: Bundled sample of 24 article-highlight pairs for offline tests; full CNN/DailyMail and XSum are fetched via scripts/fetch_datasets.py. Bundled rows are author-written; full datasets come from the HF hub. Size: 24 article-summary pairs; article mean 198 tokens, summary mean 27 tokens. Domain: English digital news articles. Challenges: Long articles exceed BART's 1024-token cap.; Webpage noise (HTML, ads) must be removed before summarisation.; Near-duplicate articles bias ROUGE if not deduplicated.; Article length varies widely across outlets.

## Team responsibilities
- **Nguyen Hoang Hieu (Ethan)**: Data collection, web scraping, and dataset organisation; Hugging Face dataset fetch scripts
- **Thai Ba Hung**: Text cleaning and preprocessing (HTML, deduplication, sentence segmentation); Exploratory data analysis and length / vocabulary statistics
- **Nguyen Quoc Dang**: Summarisation modelling (Lead-3, TextRank, BART); Hyperparameter sweeps and ablations
- **Le Nguyen Gia Binh**: ROUGE evaluation, visualisation, and qualitative review; System integration, Streamlit prototype, and final report preparation

## GitHub repository link
https://github.com/thaibahung/NLP-Project

## Submission checklist
- [ ] Private GitHub repo with collaborators `drelhaj` and `whistle-hikhi`.
- [ ] Final GitHub link included in this proposal and the PDF.
- [ ] Proposal PDF exported in 12 pt Times-compatible font, submitted to Canvas.
