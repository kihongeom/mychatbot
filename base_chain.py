from abc import ABC, abstractmethod
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI


class BaseConversationChain(ABC):

    def __init__(self, model_name="gpt-4o-mini"):
        self.model_name = model_name

    @abstractmethod
    def create_prompt(self):
        pass

    def create_model(self):
        return ChatOpenAI(model_name=self.model_name)

    def create_outputparser(self):
        return StrOutputParser()

    def create(self):
        prompt = self.create_prompt()
        model = self.create_model()
        output_parser = self.create_outputparser()
        chain = prompt | model | output_parser
        return chain
