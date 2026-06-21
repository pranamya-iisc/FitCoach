# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.
"""VLM Evaluation Code."""

import json
import os
import random
from typing import Any, List, Optional, Tuple, Union

import evaluate
import numpy as np
from datasets import Dataset
from torch import nn
from tqdm import tqdm

from src.constants import (
    FEEDBACK_BEGIN_TOKEN,
    FEEDBACK_END_TOKEN,
    INFERENCE_SPEED,
    VISION_TOKEN,
)
from src.model_wrappers import BaseVLModelWrapper


class VisionLanguageEvaluator:
    """
    Abstract base class to build evaluators for vision language datasets

    :param model:
        Model to be evaluated
    :param dataset:
        Test dataset
    """

    def __init__(self, model: Union[nn.Module, BaseVLModelWrapper], dataset: Dataset) -> None:
        self.model = model
        self.dataset = dataset

    def __len__(self) -> int:
        """
        :return:
            length of the dataset
        """
        return len(self.dataset)

    def __call__(self, **sampling_kwargs):
        """Overwrite this method in subclasses to implement custom evaluator logic"""
        raise NotImplementedError

    @staticmethod
    def _get_attention_mask(input_ids: list[int]) -> list[int]:
        """
        :param input_ids:
            List of input tokens to the model.

        :return:
            The attention mask for the language model.
        """
        attention_mask = [1] * len(input_ids)
        return attention_mask

    def _get_vision_xattn_mask(self, input_ids: list[int]) -> list[int]:
        """
        :param input_ids:
            List of input token ids to the model.
        :return:
            A list indicating locations where cross attention should be applied (VISION_TOKEN)
        """
        valid_video_indices = np.where(
            np.array(input_ids) == self.model.special_tokens_dict[VISION_TOKEN]
        )[0]
        vision_xattn_mask = np.array([0] * len(input_ids))
        vision_xattn_mask[valid_video_indices] = 1
        return vision_xattn_mask.tolist()


class InteractiveFeedbackEvaluator(VisionLanguageEvaluator):
    """Evaluator class for models that can produce interactive feedbacks"""

    def __init__(
        self,
        model: BaseVLModelWrapper,
        dataset: Dataset,
        feedbacks_save_folder: str,
        feedbacks_save_file_name: str,
    ) -> None:
        """
        :param model:
            Model to be evaluated
        :param dataset:
            Test dataset
        :param feedbacks_save_folder:
            Folder to save generated feedbacks to
        :param feedbacks_save_file_name:
            Filename for generated feedbacks file
        """
        super().__init__(model, dataset)
        # Load metrics
        self.rouge_score = evaluate.load("rouge")
        self.meteor_score = evaluate.load("meteor")
        try:
            self.bert_score = evaluate.load("bertscore")
        except (RuntimeError, FileNotFoundError, ImportError):
            self.bert_score = None

        self.mean = lambda x: sum(x) / (len(x) + 1e-12)

        # Paths to save json file with matched feedbacks
        self.feedbacks_save_path = os.path.join(
            feedbacks_save_folder, f"{feedbacks_save_file_name}.json"
        )

    def _compute_bert_scores(self, matched_feedbacks: list[tuple[str, str]]) -> list[float]:
        """
        :param matched_feedbacks:
            List of matched ground truth and predicted feedbacks.

        :return:
            List of METEOR scores.
        """
        _bert_scores = []
        if len(matched_feedbacks) > 0:
            _bert_scores = self.bert_score.compute(
                references=[x[0] for x in matched_feedbacks],
                predictions=[x[1] for x in matched_feedbacks],
                lang="en",
            )["f1"]
        return _bert_scores

    def _compute_meteor_scores(self, matched_feedbacks: list[tuple[str, str]]) -> list[float]:
        """
        :param matched_feedbacks:
            List of matched ground truth and predicted feedbacks.

        :return:
            List of METEOR scores.
        """
        _meteor_scores = []
        for matched_feedback in matched_feedbacks:
            _meteor_scores.append(
                self.meteor_score.compute(
                    references=[matched_feedback[0]], predictions=[matched_feedback[1]]
                )["meteor"]
            )
        return _meteor_scores

    def _compute_rouge_scores(self, matched_feedbacks: list[tuple[str, str]]) -> list[float]:
        """
        :param matched_feedbacks:
            List of matched ground truth and predicted feedbacks.

        :return:
            List of ROUGE scores.
        """
        _rouge_scores = []
        for matched_feedback in matched_feedbacks:
            _rouge_scores.append(
                self.rouge_score.compute(
                    references=[matched_feedback[0]], predictions=[matched_feedback[1]]
                )["rougeL"]
            )
        return _rouge_scores

    def _compute_temporal_fscore(
        self,
        gt_feedbacks: list[str],
        pred_feedbacks: list[str],
        gt_feedback_timestamps: list[float],
        pred_feedback_timestamps: list[float],
        t_f_score_running_stats: dict[str, Union[float, int]],
        eps: Optional[float] = 1e-12,
    ) -> tuple[float, list[tuple[str, str]], dict[str, Union[float, int]]]:
        """
        :param gt_feedback_timestamps:
            List of ground truth feedback timestamps.
        :param pred_feedback_timestamps:
            List of predicted feedback timestamps.
        :param gt_feedbacks:
            List of ground truth feedbacks
        :param pred_feedbacks:
            List of predicted feedbacks
        :param t_f_score_running_stats:
            Stats for computing the temporal F-score over the entire dataset.

        :return:
            Temporal F-score (float).
        """
        # Match ground truth feedbacks to predicted feedbacks (recall)
        num_matched_gt, _, matched_feedbacks, matched_timestamps = (
            self._get_temporally_aligned_feedbacks(
                gt_feedback_timestamps, pred_feedback_timestamps, gt_feedbacks, pred_feedbacks
            )
        )

        # Match predicted feedbacks to ground truth feedbacks (precision)
        _, num_matched_preds, _, _ = self._get_temporally_aligned_feedbacks(
            pred_feedback_timestamps, gt_feedback_timestamps, pred_feedbacks, gt_feedbacks
        )

        # Accumulate running stats
        t_f_score_running_stats["total_matched_gt_feedbacks"] += num_matched_gt
        t_f_score_running_stats["total_matched_pred_feedbacks"] += num_matched_preds
        t_f_score_running_stats["total_num_gt_feedbacks"] += len(gt_feedbacks)
        t_f_score_running_stats["total_num_pred_feedbacks"] += len(pred_feedbacks)

        # Compute temporal precision and recall
        precision = t_f_score_running_stats["total_matched_pred_feedbacks"] / (
            t_f_score_running_stats["total_num_pred_feedbacks"] + eps
        )
        recall = t_f_score_running_stats["total_matched_gt_feedbacks"] / (
            t_f_score_running_stats["total_num_gt_feedbacks"] + eps
        )
        f_score = 2 * ((precision * recall) / (precision + recall + eps))
        return f_score, matched_feedbacks, matched_timestamps, t_f_score_running_stats

    def _extract_pred_feedbacks(
        self, output: list[int], feats_frequency: int
    ) -> tuple[list[str], Any]:
        """
        :param output:
            List of output tokens from the model.
        :param feats_frequency:
            Number of input features per second.

        :return:
            Predicted feedback strings and their timestamps.
        """
        responses = []
        timestamps = []

        # Consider the output after the input prompt
        first_vision_token = np.where(output == self.model.special_tokens_dict[VISION_TOKEN])[0][0]
        output = output[first_vision_token:]

        # Get feedback indices
        feedback_begin_idxs = np.where(
            output == self.model.special_tokens_dict[FEEDBACK_BEGIN_TOKEN]
        )[0]
        feedback_end_idxs = np.where(output == self.model.special_tokens_dict[FEEDBACK_END_TOKEN])[
            0
        ]

        # Extract the feedback strings and timestamps.
        previous_answer_generation_time = 0
        for idx in range(min(len(feedback_begin_idxs), len(feedback_end_idxs))):
            answer_begin_idx = feedback_begin_idxs[idx]
            answer_end_idx = feedback_end_idxs[idx]

            if idx > 0:
                previous_answer_end_idx = feedback_end_idxs[idx - 1]
            else:
                previous_answer_end_idx = 1

            response = self.model.tokenizer.decode(output[answer_begin_idx + 1 : answer_end_idx])

            # Ignore empty responses
            if len(response) > 0:
                responses.append(response)

                timestep = answer_begin_idx - previous_answer_end_idx - 1
                timestep /= feats_frequency
                timestep += previous_answer_generation_time
                timestamps.append(timestep)

                previous_answer_generation_time = answer_end_idx - answer_begin_idx
                previous_answer_generation_time /= INFERENCE_SPEED
            else:
                previous_answer_generation_time = 0

        return responses, np.cumsum(timestamps).tolist()

    @staticmethod
    def _get_alignment_matrix(
        gt_feedback_timestamps: np.array,
        pred_feedback_timestamps: np.array,
        pred_feedbacks: list[str],
        tolerance: float = 3.0,
    ) -> tuple[list[int], list[int]]:
        """Perform temporal matching

        :param gt_feedback_timestamps:
            List of timestamps for ground truth feedback.
        :param pred_feedback_timestamps:
            List of timestamps for predicted feedback.
        :param pred_feedbacks:
            List of predicted feedbacks.
        :param tolerance:
            Temporal matching tolerance.

        :return:
            Indices of matching gt and pred feedbacks
        """
        matching_row_idxs, matching_col_idxs = [], []
        last_match_idx = -1
        for idx_x, x in enumerate(gt_feedback_timestamps):
            min_idx = np.argmin((pred_feedback_timestamps - x) ** 2)
            if (
                np.abs(x - pred_feedback_timestamps[min_idx]) < (tolerance / 2.0)
                and min_idx > last_match_idx
                and (min_idx not in matching_col_idxs)
                and pred_feedbacks[min_idx] != ""
            ):
                matching_row_idxs.append(idx_x)
                matching_col_idxs.append(min_idx)
                last_match_idx = min_idx
        return matching_row_idxs, matching_col_idxs

    def _get_temporally_aligned_feedbacks(
        self,
        gt_feedback_timestamps: list[float],
        pred_feedback_timestamps: list[float],
        gt_feedbacks: list[str],
        pred_feedbacks: list[str],
        tolerance: float = 3.0,
    ) -> tuple[int, int, list[tuple[str, str]]]:
        """
        Returns a list of temporally aligned feedbacks between the ground truth and predictions.

        :param gt_feedback_timestamps:
            List of ground truth feedback timestamps.
        :param pred_feedback_timestamps:
            List of predicted feedback timestamps.
        :param gt_feedbacks:
            List of ground truth feedbacks
        :param pred_feedbacks:
            List of predicted feedbacks
        :param tolerance:
            Temporal window for matching (in seconds).

        :return:
            Number of temporal matches (int) for ground truth and predicted feedbacks,
            along with the matches.
        """
        gt_feedback_timestamps = np.array(gt_feedback_timestamps)
        pred_feedback_timestamps = np.array(pred_feedback_timestamps)

        matched_feedbacks = []
        matched_idxs_gt = []
        matched_idxs_pred = []
        matching_row_idxs, matching_col_idxs = [], []
        if len(pred_feedback_timestamps) > 0:
            matching_row_idxs, matching_col_idxs = self._get_alignment_matrix(
                gt_feedback_timestamps,
                pred_feedback_timestamps,
                pred_feedbacks,
                tolerance,
            )

        matched_timestamps = []
        for match_idx, match_jdx in zip(matching_row_idxs, matching_col_idxs):
            matched_feedbacks.append((gt_feedbacks[match_idx], pred_feedbacks[match_jdx]))
            matched_timestamps.append(
                (gt_feedback_timestamps[match_idx], pred_feedback_timestamps[match_jdx])
            )
            matched_idxs_gt.append(match_idx)
            matched_idxs_pred.append(match_jdx)

        return len(matched_idxs_gt), len(matched_idxs_pred), matched_feedbacks, matched_timestamps

    @staticmethod
    def _get_video_for_episode(
        video: np.array,
        video_timestamps: np.array,
        episode_start_timestamp: Optional[float] = None,
        episode_end_timestamp: Optional[float] = None,
    ) -> np.array:
        """Returns a slice of the full video corresponding to the given time interval.

        :param video:
            Input video features for the current fitness session.
        :param video_timestamps:
            Timestamp of each video feature.
        :param episode_start_timestamp:
            Starting timestamp of the current episode.
        :param episode_end_timestamp:
            End timestamp of the current episode.

        :return:
            The video features  of the current episode.
        """
        if episode_start_timestamp is not None:
            episode_flag = np.logical_and(
                video_timestamps > episode_start_timestamp,
                video_timestamps <= episode_end_timestamp,
            )
            video = video[episode_flag]

        return video

    @staticmethod
    def _normalize_response_timestamps(
        feedback_timestamps: list[float], episode_start_timestamp: float
    ) -> list[float]:
        """
        :param feedback_timestamps: List of feedback timestamps.
        :param episode_start_timestamp: Starting timestamp of the current episode.

        :return: Predicted feedback strings and their timestamps.
        """
        return (np.array(feedback_timestamps) - episode_start_timestamp).tolist()

    @staticmethod
    def _update_save_feedbacks_dict(
        save_dict_list: List[dict],
        matched_feedbacks: List[Tuple[str, str]],
        matched_timestamps: List[Tuple[float, float]],
        meteor_scores: List[float],
        rouge_scores: List[float],
        bert_scores: List[float],
        video_id: str,
        episode_start_timestamp: float,
        episode_end_timestamp: float,
    ) -> List[dict]:
        for i, (gt_feedback, pred_feedback) in enumerate(matched_feedbacks):
            entry = {
                "video_id": video_id,
                "episode_start_timestamp": round(episode_start_timestamp, 3),
                "episode_end_timestamp": round(episode_end_timestamp, 3),
                "gt_timestamp": round(matched_timestamps[i][0], 3),
                "pred_timestamp": round(matched_timestamps[i][1], 3),
                "GT": gt_feedback,
                "Pred": pred_feedback,
                "meteor": round(meteor_scores[i], 4) if i < len(meteor_scores) else None,
                "rouge_l": round(rouge_scores[i], 4) if i < len(rouge_scores) else None,
                "bert": round(bert_scores[i], 4) if i < len(bert_scores) else None,
            }
            save_dict_list.append(entry)
        return save_dict_list

    @staticmethod
    def _dump_pred_feedbacks(save_dict_list: List[dict], save_path: str) -> None:
        os.makedirs("/".join(save_path.split("/")[:-1]), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(save_dict_list, f)

    def _print_eval_summary(
        self,
        gt_feedbacks: list[str],
        gt_feedback_timestamps: list[float],
        pred_feedbacks: list[str],
        pred_feedback_timestamps: list[float],
        t_f_score: float,
        meteor_scores: list[float],
        rouge_scores: list[float],
        bert_scores: list[float],
    ) -> None:
        """
        :param gt_feedbacks:
            List of ground truth feedbacks.
        :param gt_feedback_timestamps:
            List of ground truth feedback timestamps.
        :param pred_feedbacks:
            List of predicted feedbacks.
        :param pred_feedback_timestamps:
            List of predicted feedback timestamps.
        :param t_f_score:
            Temporal F-score.
        :param meteor_scores:
            List of METEOR scores.
        :param rouge_scores:
            List of ROUGE scores.
        :param bert_scores:
            List of BERT scores.
        """
        tqdm.write("=" * 40)
        tqdm.write("-" * 40)
        tqdm.write("GT Timestamp => GT Feedback")
        for gt_feedback, gt_feedback_timestep in zip(gt_feedbacks, gt_feedback_timestamps):
            tqdm.write(f"{gt_feedback_timestep:.2f} => {gt_feedback}")
        tqdm.write("-" * 40)
        tqdm.write("Pred Timestamp => Pred Feedback")
        for pred_feedback, pred_feedback_timestep in zip(pred_feedbacks, pred_feedback_timestamps):
            tqdm.write(f"{pred_feedback_timestep:.2f} => {pred_feedback}")
        tqdm.write("-" * 40)
        tqdm.write("Running Means ==>")
        tqdm.write(f"METEOR Score: {self.mean(meteor_scores):.3f}")
        tqdm.write(f"Rouge-L Score: {self.mean(rouge_scores):.3f}")
        if self.bert_score is not None:
            tqdm.write(f"BERT Score: {self.mean(bert_scores):.3f}")
        tqdm.write(f"Temporal F-Score: {t_f_score:.3f}")

    def __call__(self, **sampling_kwargs):
        """
        Prints the METEOR, Rouge-L, BERT and Temporal F-Scores; saves a json file with
        temporally matched feedbacks.

        :param sampling_kwargs:
            Kwargs to be passed on to the model generate method.
        """
        feats_frequency = sampling_kwargs.get("feats_frequency", 4)
        meteor_scores, bert_scores, rouge_scores = [], [], []
        t_f_score_running_stats = {
            "total_num_gt_feedbacks": 0,
            "total_num_pred_feedbacks": 0,
            "total_matched_gt_feedbacks": 0,
            "total_matched_pred_feedbacks": 0,
        }
        matched_feedbacks_to_save = []

        eval_idxs = list(range(0, len(self.dataset)))
        random.shuffle(eval_idxs)
        tqdm.write("Starting evaluation ... ")
        self.model.eval()
        for it in tqdm(eval_idxs):
            # Load data
            data = self.dataset[it]
            feat_path = data["efficientnet_features_path"]
            system_prompt = data["system"]
            gt_feedbacks = data["responses"]
            gt_feedback_timestamps = data["response_timestamps"]
            episode_start_timestamp = data["exercise_start_timestamp"]
            episode_end_timestamp = data["exercise_end_timestamp"]

            video = np.load(feat_path)
            video_timestamps = np.load(data["efficientnet_timestamps_path"])

            video = self._get_video_for_episode(
                video, video_timestamps, episode_start_timestamp, episode_end_timestamp
            )

            # Prepare inputs to the model
            input_prompt = system_prompt + VISION_TOKEN
            input_prompt = self.model.tokenizer.encode(input_prompt)
            vision_xattn_mask = self._get_vision_xattn_mask(input_prompt)
            vision_xattn_mask = [2 if tok == 1 else 0 for tok in vision_xattn_mask]

            # Generate Feedbacks
            out = self.model.generate(
                input_prompt=input_prompt,
                video=video,
                vision_xattn_mask=vision_xattn_mask,
                **sampling_kwargs,
            )

            # Process feedbacks
            pred_feedbacks, pred_feedback_timestamps = self._extract_pred_feedbacks(
                out[0], feats_frequency
            )
            gt_feedback_timestamps = self._normalize_response_timestamps(
                gt_feedback_timestamps, episode_start_timestamp
            )

            # Compute metrics
            t_f_score, matched_feedbacks, matched_timestamps, t_f_score_running_stats = (
                self._compute_temporal_fscore(
                    gt_feedbacks,
                    pred_feedbacks,
                    gt_feedback_timestamps,
                    pred_feedback_timestamps,
                    t_f_score_running_stats,
                )
            )

            episode_rouge = self._compute_rouge_scores(matched_feedbacks)
            episode_meteor = self._compute_meteor_scores(matched_feedbacks)
            episode_bert = (
                self._compute_bert_scores(matched_feedbacks)
                if self.bert_score is not None
                else [None] * len(matched_feedbacks)
            )
            rouge_scores += episode_rouge
            meteor_scores += episode_meteor
            if self.bert_score is not None:
                bert_scores += episode_bert

            # Print a running summary of metrics
            self._print_eval_summary(
                gt_feedbacks,
                gt_feedback_timestamps,
                pred_feedbacks,
                pred_feedback_timestamps,
                t_f_score,
                meteor_scores,
                rouge_scores,
                bert_scores,
            )

            # Update save dict
            video_id = os.path.splitext(os.path.basename(feat_path))[0]
            self._update_save_feedbacks_dict(
                matched_feedbacks_to_save,
                matched_feedbacks,
                matched_timestamps,
                episode_meteor,
                episode_rouge,
                episode_bert,
                video_id=video_id,
                episode_start_timestamp=episode_start_timestamp,
                episode_end_timestamp=episode_end_timestamp,
            )

        # Save feedbacks
        self._dump_pred_feedbacks(matched_feedbacks_to_save, self.feedbacks_save_path)
