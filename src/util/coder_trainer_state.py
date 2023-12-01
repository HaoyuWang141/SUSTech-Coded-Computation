from dataclasses import dataclass


@dataclass
class CoderTrainerState:
    epoch: int
    encoder_state_dict: dict
    decoder_state_dict: dict
    encoder_optimizer_state_dict: dict
    decoder_optimizer_state_dict: dict
    loss_history: list
