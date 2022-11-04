import evaluate
from evaluate.evaluation_suite import SubTask, Preprocessor
from datasets import Dataset

"""
Notes/Questions:
1. Overriding the 'task' (init argument) for Evaluator objects isn't possible through this
2. As part of data preparation, I might need to flatten or filter the dataset
3. For SubTasks that share the same data + task_type, is there a way to combine the evaluation for efficiency?
4. Is it possible to select nested features in the input columns?
5. Can the evaluator be used to pass in batches? Instead of one-by-one"
6. 
"""


class ToxicityPreprocessor(Preprocessor):
    def run(self, dataset: Dataset) -> Dataset:
        dataset = dataset.flatten()
        return dataset


class Suite(evaluate.EvaluationSuite):

    def __init__(self, name):
        super().__init__(name)
        # self.preprocessor = lambda x: {"text": x["text"].lower()}
        self.suite = [
            SubTask(
                task_type="text-generation",
                data="allenai/real-toxicity-prompts",
                split="train[:10]",  # TODO: Full dataset...
                data_preprocessor=ToxicityPreprocessor(),
                args_for_task={
                    "metric": "toxicity",
                    "input_column": "prompt.text",
                }
            ),
        ]
