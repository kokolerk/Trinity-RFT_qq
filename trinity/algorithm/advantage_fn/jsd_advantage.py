# -*- coding: utf-8 -*-
"""Jensen-Shannon Divergence (JSD) advantage computation.

JSD(P||Q) = lambda * KL(P||M) + (1-lambda) * KL(Q||M), where M = (P+Q)/2
When lambda=0.5, this gives the standard symmetric JSD.

For advantage function, we compute the per-token contribution to JSD
and use it to guide the student model towards the teacher model.
"""

from typing import Dict, Tuple

import torch
from verl import DataProto

from trinity.algorithm.advantage_fn.advantage_fn import AdvantageFn


class JSDAdvantage(AdvantageFn):
    """Advantage function using Jensen-Shannon Divergence.

    Computes JSD using the correct formula:
    JSD(P||Q) = lambda * KL(P||M) + (1-lambda) * KL(Q||M), where M = (P+Q)/2
    When lambda=0.5, this gives the standard symmetric JSD.

    For advantage computation, we use the gradient direction of JSD
    with respect to the student model, which guides the student towards
    minimizing JSD with the teacher.

    The teacher_logprobs should be stored in Experience.teacher_logprobs
    by the workflow during exploration.
    """

    def __init__(self, lambda_coef: float = 0.5, kl_coef: float = 1.0) -> None:
        """Initialize JSD advantage function.

        Args:
            lambda_coef: Weight for mixing KL(P||M) and KL(Q||M) in JSD.
                        JSD = lambda * KL(P||M) + (1-lambda) * KL(Q||M).
                        Standard symmetric JSD uses 0.5. Range: [0, 1].
            kl_coef: Overall scaling coefficient for advantages.
        """
        self.lambda_coef = lambda_coef
        self.kl_coef = kl_coef

    def _js_divergence_per_token(
        self, student_logprobs: torch.Tensor, teacher_logprobs: torch.Tensor
    ) -> torch.Tensor:
        """Compute JSD per token: JSD(P||Q) = lambda * KL(P||M) + (1-lambda) * KL(Q||M), M = (P+Q)/2.

        This follows the reference implementation:
        JSD(P||Q) = lambda * KL(P||M) + (1-lambda) * KL(Q||M), where M = (P+Q)/2
        When lambda=0.5, this gives the standard symmetric JSD.

        Args:
            student_logprobs: Student log probabilities [batch, seq]
            teacher_logprobs: Teacher log probabilities [batch, seq]

        Returns:
            JSD per token [batch, seq]
        """
        # M = (P+Q)/2, so log(M) = logsumexp([student_logprobs, teacher_logprobs]) - log(2)
        m = torch.logsumexp(torch.stack([student_logprobs, teacher_logprobs]), dim=0) - torch.log(
            torch.tensor(2.0, device=student_logprobs.device)
        )

        # KL(P||M) per token: (p/m) * log(p/m) = exp(student_logprobs - logm) * (student_logprobs - logm)
        kl_p_m = torch.exp(student_logprobs - m) * (student_logprobs - m)

        # KL(Q||M) per token: (q/m) * log(q/m) = exp(teacher_logprobs - logm) * (teacher_logprobs - logm)
        kl_q_m = torch.exp(teacher_logprobs - m) * (teacher_logprobs - m)

        # JSD = lambda * KL(P||M) + (1-lambda) * KL(Q||M)
        jsd = self.lambda_coef * kl_p_m + (1.0 - self.lambda_coef) * kl_q_m

        return jsd

    def __call__(self, exps: DataProto, **kwargs) -> Tuple[DataProto, Dict]:
        """Compute advantages using JSD.

        Advantages are computed directly from JSD: advantages = -kl_coef * JSD
        Since we want to minimize JSD, we use negative JSD as advantage.
        Lower JSD (better alignment with teacher) → higher advantage.
        The advantage guides the policy gradient to reduce JSD.

        Args:
            exps: DataProto containing:
                - old_log_probs: student's sampling logprobs [batch, seq]
                - teacher_logprobs: teacher's logprobs [batch, seq]
                - response_mask: mask for response tokens [batch, seq]

        Returns:
            exps: DataProto with advantages and returns added
            metrics: Dict with jsd and advantage statistics
        """
        metrics = {}

        old_log_probs = exps.batch["old_log_probs"]  # student sampling logprobs
        teacher_log_probs = exps.batch["teacher_logprobs"]
        response_mask = exps.batch["response_mask"]

        # Compute JSD per token
        jsd_per_token = self._js_divergence_per_token(old_log_probs, teacher_log_probs)
        
        # For advantage function, use JSD directly
        # Since we want to minimize JSD, we use negative JSD as advantage
        # This means: lower JSD (better alignment) → higher advantage
        # The advantage guides the policy gradient to reduce JSD
        advantages = -self.kl_coef * jsd_per_token

        # Apply mask
        advantages = advantages * response_mask

        exps.batch["advantages"] = advantages
        exps.batch["returns"] = advantages.clone()

        # JSD metrics
        jsd_sum = (jsd_per_token * response_mask).sum(dim=-1)
        metrics["jsd/mean"] = jsd_sum.mean().item()
        metrics["jsd/std"] = jsd_sum.std().item() if jsd_sum.numel() > 1 else 0.0

        metrics["advantages/mean"] = advantages.sum(dim=-1).mean().item()

        return exps, metrics

    @classmethod
    def default_args(cls) -> Dict:
        return {"lambda_coef": 0.5, "kl_coef": 1.0}
