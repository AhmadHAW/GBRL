from .agent import Agent

from typing import Tuple

class DummyAgent(Agent):
    '''
    The DummyAgent is a mock up agent, without any intelligent backend.
    The expected response has to be passed with every textual entailmen operation.
    We are using this agent to formulate expectations and get a feeling for the user interaction.
    '''

    def __init__(self, log_file = None, feedback_file = None, layout = "planar", font_size = 12, node_size = 300):
        '''
        See Agent constructor.
        '''
        super().__init__(log_file, feedback_file, layout, font_size, node_size)

    
    def _get_response(self, query, **kwargs) -> Tuple[str,str]:
        '''
        In addition to the query, this method expects the key value pair {"dummy_response":"Some expected response."} in kwargs. 
        '''
        if "dummy_response" not in kwargs:
            raise Exception("Expecting a dummy response passed with the textual entailment operation, \
                             like: {'dummy_response':'Some expected response.'}")
        return kwargs["dummy_response"], kwargs["dummy_response"] #There is no difference between the original response and its cleaned version.
