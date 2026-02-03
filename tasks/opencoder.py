import re
from datasets import load_dataset
from tasks.common import Task

class OpenCoder(Task):

    def __init__(self, subset, split, **kwargs):
        super().__init__(**kwargs)
        assert split in ["train", "test"], "OpenCoder split must be train|test"
        # Load the OpenCoder dataset
        self.ds = load_dataset("OpenCoder-LLM/opc-sft-stage2", subset, split=split).shuffle(seed=42)

    @property
    def eval_type(self):
        return 'generative'

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        """Get a single problem from the dataset."""
        row = self.ds[index]
        question = row['instruction']  # Assuming 'instruction' contains the question
        answer = row['output']  # Assuming 'output' contains the full solution and answer

        # For OpenCoder, the answer is typically a string, not a tool call
        messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},  # Simple string, no tool calls
        ]
        conversation = {
            "messages": messages,
        }
        return conversation

    def evaluate(self, conversation, assistant_response):
        """
        Compare the predicted answer to the ground truth answer.
        Returns 1 if correct, 0 otherwise.
        """
        assert isinstance(assistant_response, str), "Assuming simple string response for now"
        assistant_message = conversation['messages'][-1]
        assert assistant_message['role'] == "assistant", "Last message must be from the Assistant"
        ref_answer = assistant_message['content']
        pred_answer = assistant_response
        # Compare the predicted answer to the reference answer
        is_correct = int(normalize_answer(pred_answer) == normalize_answer(ref_answer))
        return is_correct

    def reward(self, conversation, assistant_response):
        """Used during RL. Re-use the evaluation logic."""
        is_correct = self.evaluate(conversation, assistant_response)
        return float(is_correct)

# Helper function to normalize answers for comparison
def normalize_answer(answer):
    """Normalize the answer for comparison (e.g., strip whitespace, lowercase, etc.)."""
    return answer.strip().lower()
