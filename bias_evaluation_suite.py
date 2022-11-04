import evaluate
from evaluate.evaluation_suite import SubTask, Preprocessor
from evaluate import TextGenerationEvaluator
from datasets import Dataset

"""
Notes/Questions:
1. Overriding the 'task' (init argument) for Evaluator objects isn't possible through this
2. As part of data preparation, I might need to flatten or filter the dataset
3. For SubTasks that share the same data + task_type, is there a way to combine the evaluation for efficiency?
4. Is it possible to select nested features in the input columns? (I guess by flattening?)
5. Can the evaluator be used to pass in batches? Instead of one-by-one"
    How can batch_size be set for the pipeline? I guess the generation_kwargs?
6. Metric arguments need to be passed down (e.g. the config_name for HONEST is required to specify the language)
    In general any of the __init__ arguments for EvaluationModule, really
7. Different metrics expect predictions in different formats (e.g. toxicity (strings) vs. HONEST (lists)). They're both
    used for the text_generation evaluator though! So somehow either the evaluator needs to know what data format is
    expected by the metric, or we need to be able to pass a custom post-processor.
8. Is it possible to see a progress bar for each evaluation task?
"""


class ToxicityPreprocessor(Preprocessor):
    def run(self, dataset: Dataset) -> Dataset:
        dataset = dataset.flatten()
        return dataset


class HonestEvaluator(TextGenerationEvaluator):
    def predictions_processor(self, predictions, label_mapping):
        return {
            "predictions": [[pred[f"{self.PREDICTION_PREFIX}_text"].split(" ")[0] for pred in predictions[0]]]
        }


class RegardEvaluator(TextGenerationEvaluator):
    def predictions_processor(self, predictions, label_mapping):
        return {
            "data": [pred[f"{self.PREDICTION_PREFIX}_text"].split(" ")[0] for pred in predictions[0]]
        }


class Suite(evaluate.EvaluationSuite):

    def __init__(self, name):
        super().__init__(name)
        self.suite = [
            SubTask(
                task_type="text-generation",
                name="toxicity",
                data="allenai/real-toxicity-prompts",
                split="train",  # TODO: Full dataset...
                data_preprocessor=ToxicityPreprocessor(),
                args_for_task={
                    "metric": "toxicity",
                    "input_column": "prompt.text",
                    "generation_kwargs": {
                        # "do_sample": False,
                        "max_length": 60
                    }
                }
            ),
            SubTask(
                task_type="text-generation",
                name="HONEST",
                evaluator=HonestEvaluator(),
                data="MilaNLProc/honest",
                subset="en_binary",
                split="honest[:10]",
                data_preprocessor=lambda x: {"text": x["template_masked"][:-4]},
                args_for_task={
                    "metric": "honest",
                    "metric_init_kwargs": {
                      "config_name": "en"
                    },
                    "input_column": "text",
                    "generation_kwargs": {
                        "return_full_text": False
                    },
                }
            ),
            SubTask(
                task_type="text-generation",
                name="regard",
                evaluator=RegardEvaluator(),
                data="MilaNLProc/honest",
                subset="en_binary",
                split="honest[:10]",
                data_preprocessor=lambda x: {"text": x["template_masked"][:-4]},
                args_for_task={
                    "metric": "regard",
                    "input_column": "text",
                }
            ),
        ]
