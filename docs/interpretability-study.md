# Research plan: can an agent reproduce a causal interpretability analysis?

**Completed foundation:** matched-prompt activation interventions and Distributed Alignment Search (DAS), with held-out comparisons that measure whether a changed prediction persists after a follow-up and whether unrelated attributes remain stable. The proposed work below is a separate next experiment; it does not report completed agent or fine-tuning results.

## Question

Can a tool-using agent reproduce a fixed, human-reviewed causal finding when the same model is evaluated before and after fine-tuning? A useful finding must predict an intervention result on unseen examples. A coherent explanation or a probe that reads a feature is not sufficient evidence that the model uses it.

## Experimental design

1. Start with base and fine-tuned checkpoints of the same architecture and tokenizer. Save exact checkpoint IDs, fine-tuning data and configuration, prompts, decoding settings, and seeds. Pre-register a narrow change to test, such as following a stated fact when a distractor is introduced.
2. Run each checkpoint on identical clean/corrupted prompt pairs. Record clean accuracy and the difference between target and distractor logits. Include matched no-change prompts and unrelated attributes as controls. Use examples that both checkpoints answer correctly as a matched subset, and also report results on the full dataset to expose selection effects.
3. Patch clean activations into the corresponding corrupted run **within each checkpoint**. Compare layer/token maps and held-out recovery. This avoids assuming that the same vector has the same meaning in different checkpoint spaces. Report paired uncertainty, failure examples, and changes in isolation and follow-up persistence.
4. Give an agent bounded tools: inspect logits, run a specified patch, and score a held-out hypothesis. Limit allowed checkpoints, intervention sites, and tool calls. Require a structured report with evidence and predicted effects before releasing held-out results.
5. Compare the agent against a fixed scripted scan at equal patch budgets, a transcript-only agent, and a human-reviewed reference. Record validated findings per tool call, agreement on effect direction, intervention-effect error, spurious findings on controls, and time/cost. Keep the test split hidden from the agent's search.

## What would count as progress?

An agent that identifies a post-fine-tuning change must give a testable prediction (for example, patching a specific site recovers the target in the adapted checkpoint but not the base checkpoint), and the effect must replicate on held-out pairs. A fixed scan may win on accuracy or cost; report that comparison. Changes in visible behavior alone do not establish a different internal mechanism.

## Research that informs this design

- [CHIVE (2026)](https://arxiv.org/abs/2608.16747) uses an agent to find and test explanations with counterfactual prompt edits. The authors report that the interpretability tools they evaluated did not improve the agent's predictions of counterfactual behavior. This motivates comparing a tool-using agent to a transcript-only agent and grading predictions instead of explanation fluency.
- [Control-Diverse Reinforcement Fine-Tuning (2026 preprint)](https://arxiv.org/abs/2608.08224) distinguishes a component's activation from its control over gains after post-training. This motivates recording activation changes and causal intervention effects separately.
- [RAVEL](https://arxiv.org/abs/2402.17700) uses controlled interventions to test whether an attribute can be changed while other attributes remain intact. The study design uses that distinction for isolation and includes an added follow-up test for persistence.
- [Circuit Tracing](https://transformer-circuits.pub/2025/attribution-graphs/methods.html) compares a more interpretable replacement model with the underlying model and tests the proposed mechanisms with perturbations. Here, an agent's proposed circuit is treated as a hypothesis to be checked on the original checkpoint.
