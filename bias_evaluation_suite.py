import evaluate
from evaluate.evaluation_suite import SubTask, Preprocessor
from evaluate import TextGenerationEvaluator
from datasets import Dataset, DatasetDict, concatenate_datasets
from typing import Tuple, Any, Dict
import pandas as pd

"""
Notes/Questions:
1. Overriding the 'task' (init argument) for Evaluator objects isn't possible through the current evaluation suite API
2. As part of data preparation, I might need to flatten or filter the dataset
3. For SubTasks that share the same data + task_type, is there a way to combine the evaluation for efficiency?
4. Is it possible to select nested features for the input columns? (A: I guess by flattening?)
5. Can the evaluator be used to pass in batches? Instead of one-by-one
    How can batch_size be set for the pipeline? I guess the generation_kwargs?
6. Metric arguments need to be passed down (e.g. the config_name for HONEST is required to specify the language)
    In general any of the __init__ arguments for EvaluationModule, really
    As well as compute-time arguments
7. Different metrics expect predictions in different formats (e.g. toxicity (strings) vs. HONEST (lists)). They're both
    used for the text_generation evaluator though! So somehow either the evaluator needs to know what data format is
    expected by the metric, or we need to be able to pass a custom post-processor.
8. Is it possible to see a progress bar for each evaluation task?
9. Inputs get passed to the pipeline as DatasetColumn, which is not Dataset
     is_dataset = Dataset is not None and isinstance(inputs, Dataset) --> false
     this doesn't cause anything weird because it's caught by is_list below it, but still worth noting
10. How can I handle very large evaluations without blowing up memory? e.g. with toxicity, the benchmark dataset is
      quite large, and the Evaluators want all the predictions BEFORE computing the metric. Is that realistic? Is there
      a way to do chunked metric calculations? Is that even beneficial at all?
11. The Regard metric needs to receive data & references, which are basically disaggregated results from the pipeline
      run. In order to do that, I have to override the predictions_processor, but it ALSO needs the original data so
      that I can do the disaggregation... HOLD UP! I can actually just do the aggregation AFTER the metric calculation,
      in my custom evaluator by overriding .compute()
12. (BIG, possibly bug?) PIPELINE_KWARGS in the various evaluators is a class variable, which means that it's shared
      among instances of the same evaluator type. This is problematic because e.g. in the evaluation suites it should
      be possible to use the same evaluator type (e.g. text generation) for multiple tasks, which can require completely
      different PIPELINE_KWARGS. In my case I ran into this with num_return_sequences. IMO I can't see any good reason
      for class variables in the evaluators.
"""

# https://colab.research.google.com/drive/1-HDJUcPMKEF-E7Hapih0OmA1xTW2hdAv#scrollTo=1Uk8NROQ3l-k


class ToxicityPreprocessor(Preprocessor):
    def run(self, dataset: Dataset) -> Dataset:
        dataset = dataset.flatten()
        dataset = dataset.shuffle().select(range(100))  # TODO: Temporary, for replicating Colab results
        return dataset


class BoldPreprocessor(Preprocessor):
    def run(self, dataset: Dataset) -> Dataset:
        """
        TODO: To replicate colab results
        Sample 50 American_actresses + 50 American_actors, and only select the first prompt for each
        """
        shuffled_dataset = dataset.shuffle()
        female_bold = shuffled_dataset.filter(lambda x: x["category"] == "American_actresses").select(range(50))
        male_bold = shuffled_dataset.filter(lambda x: x["category"] == "American_actors").select(range(50))
        dataset = concatenate_datasets([female_bold, male_bold])

        # Index needed for selecting the subgroups later
        dataset = dataset.map(lambda x, idx: {"prompt": x["prompts"][0], "index": idx}, with_indices=True)

        return dataset


class RegardEvaluator(TextGenerationEvaluator):
    def __init__(self, compare_by, label1, label2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.compare_by = compare_by
        self.label1 = label1
        self.label2 = label2

    def predictions_processor(self, predictions, label_mapping):
        # Regard expects the field to be called "data"
        return {"data": [pred[0][f"{self.PREDICTION_PREFIX}_text"] for pred in predictions]}

    def compute(self, *args, **kwargs) -> Tuple[Dict[str, float], Any]:
        # NOTE: I implemented *basically* the same thing for HONEST (two computations), but in a slightly different way
        #       Compare the two to see which one you prefer...

        result = super().compute(*args, **kwargs)

        result["regard"] = list(map(lambda r: {rs["label"]: rs["score"] for rs in r}, result["regard"]))

        data = kwargs["data"]
        group1 = data.filter(lambda x: x[self.compare_by] == self.label1)
        group2 = data.filter(lambda x: x[self.compare_by] == self.label2)

        results_df = pd.DataFrame(result["regard"])
        result["regard"] = {
            "regard_difference": dict(results_df.iloc[group1["index"]].mean() / results_df.iloc[group2["index"]].mean())
        }

        return result


class HonestPreprocessor(Preprocessor):
    def __init__(self, compare_by, label1, label2):
        self.compare_by = compare_by
        self.label1 = label1
        self.label2 = label2

    def run(self, dataset: Dataset) -> DatasetDict:
        shuffled_dataset = dataset.shuffle()
        group1 = shuffled_dataset.filter(lambda x: x[self.compare_by].startswith(self.label1)).select(range(50))
        group2 = shuffled_dataset.filter(lambda x: x[self.compare_by].startswith(self.label2)).select(range(50))

        # TODO: Might want to abstract out hard-coded values ("text", "template_masked")
        dataset = DatasetDict({"group1": group1, "group2": group2}).map(lambda x: {"text": x["template_masked"][:-4]})

        return dataset


class HonestEvaluator(TextGenerationEvaluator):
    # TODO: Pretty similar to the Regard evaluator...

    def predictions_processor(self, predictions, label_mapping):
        return {
            "predictions": [p[f"{self.PREDICTION_PREFIX}_text"].split() for ps in predictions for p in ps],
        }

    def compute(self, *args, **kwargs) -> Tuple[Dict[str, float], Any]:
        data1 = kwargs["data"]["group1"]
        data2 = kwargs["data"]["group2"]

        results1 = super().compute(*args, **{**kwargs, "data": data1})
        results2 = super().compute(*args, **{**kwargs, "data": data2})

        return {"group1": results1, "group2": results2}


class Suite(evaluate.EvaluationSuite):

    def __init__(self, name):
        super().__init__(name)
        self.suite = [
            SubTask(
                task_type="text-generation",
                name="toxicity",
                data="allenai/real-toxicity-prompts",
                split="train",
                data_preprocessor=ToxicityPreprocessor(),
                metric_kwargs={
                    "aggregation": "ratio"
                },
                args_for_task={
                    "metric": "toxicity",
                    "input_column": "prompt.text",
                    "generation_kwargs": {
                        "max_length": 50,
                        "do_sample": False,
                        # TODO: This is specific to GPT-2 though, because it's not in the model confic...
                        # Should find a way to make this dynamic, when it's necessary
                        # "pad_token_id": 50256,
                        "return_full_text": False,
                    }
                }
            ),
            SubTask(
                task_type="text-generation",
                name="regard",
                evaluator=RegardEvaluator(
                    compare_by="category",
                    label1="American_actresses",
                    label2="American_actors"
                ),
                data="AlexaAI/bold",
                split="train",
                data_preprocessor=BoldPreprocessor(),
                args_for_task={
                    "metric": "regard",
                    "input_column": "prompt",
                    "generation_kwargs": {
                        "max_length": 50,  # TODO: Need to make this dynamic, same with the one in HONEST below
                        "do_sample": False,
                        # "pad_token_id": 50256,
                        "return_full_text": False,
                    },
                }
            ),
            SubTask(
                task_type="text-generation",
                name="HONEST",
                evaluator=HonestEvaluator(),
                data="MilaNLProc/honest",
                subset="en_queer_nonqueer",
                split="honest",
                data_preprocessor=HonestPreprocessor(
                    compare_by="category",
                    label1="queer",
                    label2="nonqueer"
                ),
                args_for_task={
                    "metric": "honest",
                    "metric_init_kwargs": {
                        "config_name": "en"
                    },
                    "input_column": "text",
                    "generation_kwargs": {
                        "max_length": 50,  # TODO: Needs to be dynamic, e.g. len(tokenizer(prompt)['input_ids'])+10
                        "num_return_sequences": 20,
                        # "pad_token_id": 50256,
                        "return_full_text": False,
                    },
                }
            ),
        ]
