import numpy as np


class Evaluate:
    """
    Evaluate model predictions using a specified metric.

    """

    def __init__(self, tokenizer, metric):
        self.tokenizer = tokenizer
        self.metric = metric

    def _postprocess_text(self, preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]

        return preds, labels

    def compute_metrics(self, eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds, decoded_labels = self._postprocess_text(
            decoded_preds, decoded_labels
        )

        result = self.metric.compute(
            predictions=decoded_preds, references=decoded_labels
        )
        result = {"bleu": result["score"]}

        prediction_lens = [
            np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in preds
        ]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}

        return result


if __name__ == "__main__":
    pass
