# RGQALE
Review Guided Question Answering in Live Stream E-commerce 

## Data
### [AmazonQA](https://github.com/amazonqa/amazonqa)

The dataset is .jsonl format, where each line in the file is a json string that corresponds to a question, existing answers to the question and the extracted review snippets (relevant to the question).

Each json string has many fields. Here are the fields that the QA training pipeline uses:

- **questionText**: String. The question.
- **questionType**: String. Either "yesno" for a boolean question, or "descriptive" for a non-boolean question.
- **review_snippets**: List of strings. Extracted review snippets relevant to the question (at most ten).
- **answers**: List of dicts, one for each answer. Each dict has the following fields.
- **answerText**: String. The text for the answer.
- **answerType**: String. Type of the answer.
- **helpful**: List of two integers. The first integer indicates the number of uses who found the answer helpful. The second integer indicates the total number of responses.

Here are some other fields that we use for evaluation and analysis:

- **asin**: String. Unique product ID for the product the question pertains to.
- **qid**: Integer. Unique question id for the question (in the entire dataset).
- **category**: String. Product category.
- **top_review_wilson**: String. The review with the highest wilson score.
- **top_review_helpful**: String. The review voted as most helpful by the users.
- **is_answerable**: boolean. Output of the answerability classifier indicating whether the question is answerable using the review snippets.
- **top_sentences_IR**: List of strings. A list of top sentences (at most 10) based on IR score with the question.


