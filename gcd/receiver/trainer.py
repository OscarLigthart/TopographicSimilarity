import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReceiverTrainer(nn.Module):
    """
    This class trains just the receiver. The reason for the existence of this class
    is to check model performance when we insert a perfect compositional language
    We don't need a sender if the language is known, hence we just train the receiver
    """
    def __init__(self, receiver):
        super().__init__()

        self.receiver = receiver

    def forward(self, messages, target, distractors):

        # get batch size
        batch_size = target.shape[0]

        # put targets and distractors on device
        target = target.to(device)
        distractors = [d.to(device) for d in distractors]

        # run receiver
        h_r, h_rnn_r = self.receiver(messages=messages)

        # reshape targets
        target = target.view(batch_size, 1, -1)
        r_transform = h_r.view(batch_size, -1, 1)

        # get target score
        target_score = torch.bmm(target, r_transform).squeeze()  # scalar
        all_scores = torch.zeros((batch_size, 1 + len(distractors)))
        all_scores[:, 0] = target_score

        # calculate loss
        loss = 0
        for i, d in enumerate(distractors):
            d = d.view(batch_size, 1, -1)
            d_score = torch.bmm(d, r_transform).squeeze()
            all_scores[:, i + 1] = d_score
            loss += torch.max(
                torch.tensor(0.0, device=device), 1.0 - target_score + d_score
            )

        # Calculate accuracy
        all_scores = torch.exp(all_scores)
        _, max_idx = torch.max(all_scores, 1)

        accuracy = max_idx == 0
        accuracy = accuracy.to(dtype=torch.float32)

        if self.training:
            return (torch.mean(loss), torch.mean(accuracy), messages)
        else:
            return (
                torch.mean(loss),
                torch.mean(accuracy),
                messages,
                h_r.detach(),
                h_rnn_r.detach()
            )
