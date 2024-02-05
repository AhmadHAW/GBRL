from .agent import Agent

from typing import Tuple

INIT_PROMPT = """I will give you a set of instructions in a certain style in tripple quotes. Make sure you respond only in the syntax i expect. I will give you multiple examples and expeced responses with an explanation.
example 1: '''<s1><:>I love my dog.<;><s2><:>I don't own a dog.<;><e1><:><rel1><;><s1><e1><s2><;>'''
example 1 expected response: '''<rel1><:><con><;>'''
example 1 explanation: Two nodes <s1> and <s2> are defined and we ask you to solve for the textual entailment relationship <e1> between those nodes with its placeholder <rel1>. Your task is to predict the placeholder <rel1>, which can be either <ent> for entailment, <con> for contradiction,<neu> for neutral, <nent> for not entailment, <ncon> for not contradiction or <nneu> for not neutral. The statement <s1> strongly suggest the ownership which contradicts <s2>, thats why <rel1> can be filled with <con>.
example 2: '''<s1><:>I is raining on the street.<;><s2><:><t2><;><s1><ent><s2><;>'''
example 2 expected response: '''<t2><:>The street is wet.<;>'''
example 2 explanation: Instead of solving for the relationship between two nodes, this task prompts you to solve for a node <s2> itself, that fits the given relationships. A possible answer could be: "The street is wet.", because it is almost impossible for the street not to be wet if it's raining. The placeholder <t2> suggests you should try to fill this node with a statement.
example 3: '''<s1><:>A lumberjack is working in the woods.<;><s2><:><t2><;><s3><:>A tree fell to the ground<;><s1><ent><s2><;><s2><ent><s3><;><s1><ent><s3><;>'''
example 3 expected response: '''<t2><:>The lumberjack fell a tree.<;>'''
example 3 explanation: In this example the task is to produce a statement that fill the position for <t2> with the placeholder <t2>, so that all relations between the statements are considered true.
Say begin if you understood."""


class ChatGPTAgent(Agent):
    '''
    The ChatGPTAgent is using Chat-GPT as LLM to solve the textual entailment tasks given.
    For that the user will have to copy and paste an initial prompt for zero-shot prompting
    and all following queries. At any point, the Chat-GPT conversation can be restarted with
    the initial prompt and the following queries.
    '''

    def __init__(self, log_file = None, feedback_file = None, layout = "planar", font_size = 12, node_size = 300):
        '''
        See Agent constructor.
        '''
        super().__init__(log_file, feedback_file, layout, font_size, node_size)
        self._prompt_init()

    def reset(self):
        '''
        This reset adds the print of the initial prompt for zero-shot prompting to its call.
        '''
        self.G.clear()
        self.logs = [] if self.log_file else None
        self._prompt_init()


    def _prompt_init(prompt):
        print("Pass the following prompt to Chat-GPT:")
        print(INIT_PROMPT)

    
    def _get_response(self, query, **kwargs) -> Tuple[str,str]:
        '''
        The response is produced by Chat-GPT when passed.
        The manual copy-paste can be exchanged by an API-access.
        '''
        print("Pass the following text to ChatGPT and return with the response.")
        print(f"'''{query}'''")
        response = input("Chat-GPT Response: ")
        output = response[3:-3]
        return output, response