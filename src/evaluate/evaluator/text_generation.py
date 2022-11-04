# Copyright 2022 The HuggingFace Evaluate Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Callable, Dict, Optional, Tuple, Union

from datasets import Dataset
from typing_extensions import Literal

from ..module import EvaluationModule
from ..utils.file_utils import add_start_docstrings
from .base import EVALUATOR_COMPUTE_RETURN_DOCSTRING, EVALUTOR_COMPUTE_START_DOCSTRING, Evaluator
from .utils import DatasetColumn


TASK_DOCUMENTATION_KWARGS = r"""
        input_column (`str`, defaults to `"text"`):
            the name of the column containing the input text in the dataset specified by `data`.
        label_column (`str`, defaults to `"label"`):
            the name of the column containing the labels in the dataset specified by `data`.
        generation_kwargs (`Dict`, *optional*, defaults to `None`):
            The generation kwargs are passed to the pipeline and set the text generation strategy.
"""


class TextGenerationEvaluator(Evaluator):
    """
    Text generation evaluator.
    This Text generation evaluator can currently be loaded from [`evaluator`] using the default task name
    `text-generation`.
    Methods in this class assume a data format compatible with the [`TextGenerationPipeline`].
    """

    PREDICTION_PREFIX = "generated"
    PIPELINE_KWARGS = {}

    def __init__(self, task="text-generation", default_metric_name=None):
        super().__init__(task, default_metric_name=default_metric_name)

    def predictions_processor(self, predictions, label_mapping):
        return {"predictions": [pred[f"{self.PREDICTION_PREFIX}_text"] for pred in predictions[0]]}

    def prepare_data(self, data: Dataset, input_column: str, label_column: str):
        """
        Prepare data.

        Args:
            data (`Dataset`): Specifies the dataset we will run evaluation on.
            input_column (`str`, defaults to `"text"`):
                the name of the column containing the text feature in the dataset specified by `data`.
        Returns:
            `dict`:  metric inputs.
            `list`:  pipeline inputs.
        """

        self.check_required_columns(data, {"input_column": input_column })

        return {}, DatasetColumn(data, input_column)


    @add_start_docstrings(
        EVALUTOR_COMPUTE_START_DOCSTRING, TASK_DOCUMENTATION_KWARGS, EVALUATOR_COMPUTE_RETURN_DOCSTRING
    )
    def compute(
        self,
        model_or_pipeline: Union[
            str, "Pipeline", Callable, "PreTrainedModel", "TFPreTrainedModel"  # noqa: F821
        ] = None,
        data: Union[str, Dataset] = None,
        subset: Optional[str] = None,
        split: Optional[str] = None,
        metric: Union[str, EvaluationModule] = None,
        metric_init_kwargs: Optional[Dict] = None,
        tokenizer: Optional[Union[str, "PreTrainedTokenizer"]] = None,  # noqa: F821
        strategy: Literal["simple", "bootstrap"] = "simple",
        confidence_level: float = 0.95,
        n_resamples: int = 9999,
        device: int = None,
        random_state: Optional[int] = None,
        input_column: str = "text",
        generation_kwargs: dict = None,
    ) -> Tuple[Dict[str, float], Any]:
        """
        Examples:
        ```python
        >>> from evaluate import evaluator
        >>> from datasets import load_dataset
        >>> task_evaluator = evaluator("text2text-generation")
        >>> data = load_dataset("cnn_dailymail", "3.0.0", split="validation[:40]")
        >>> results = task_evaluator.compute(
        >>>     model_or_pipeline="facebook/bart-large-cnn",
        >>>     data=data,
        >>>     input_column="article",
        >>>     metric="rouge",
        >>> )
        ```
        """

        if generation_kwargs is not None:
            self.PIPELINE_KWARGS.update(generation_kwargs)

        result = super().compute(
            model_or_pipeline=model_or_pipeline,
            data=data,
            subset=subset,
            split=split,
            metric=metric,
            metric_init_kwargs=metric_init_kwargs,
            tokenizer=tokenizer,
            strategy=strategy,
            confidence_level=confidence_level,
            n_resamples=n_resamples,
            device=device,
            random_state=random_state,
            input_column=input_column,
        )

        return result
