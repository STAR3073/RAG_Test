from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI


def create_english_conversation_chain(model_name="gpt-4o-mini"):
    # LCEL 문법 활용 -> Chain 을 생성
    template = """당신은 영어를 가르치는 10년차 영어 선생님입니다. 주어진 상황에 맞는 영어 회화를 작성해 주세요.
    양식은 Example을 참고하여 작성해 주세요.

    #상황:
    {question}

    #Example:
    **영어 회화**
    - Customer: Hi there! I would like to order a pizza, please.
    - Staff: Sure! What size would you like?
    - Customer: I’ll have a large pizza, please.
    - Staff: Great! What toppings do you want?
    ...

    **한글 해석**
    - 고객: 안녕하세요! 피자를 주문하고 싶어요.
    - 직원: 물론입니다! 어떤 사이즈로 주문하시겠어요?
    - 고객: 대형 피자로 주세요.
    - 직원: 좋습니다! 어떤 토핑을 원하시나요?
    ...
    """

    # 프롬프트 템플릿을 이용하여 프롬프트를 생성합니다.
    prompt = PromptTemplate.from_template(template)

    # ChatOpenAI 챗모델을 초기화합니다.
    model = ChatOpenAI(model_name=model_name)

    # 문자열 출력 파서를 초기화합니다.
    output_parser = StrOutputParser()

    # Chain 을 생성합니다.
    english_chain = prompt | model | output_parser
    
    return english_chain



from abc import ABC, abstractmethod


class BaseConversationChain(ABC):

    @abstractmethod
    def create_prompt(self):
        pass
    
    @abstractmethod
    def create_model(self):
        return ChatOpenAI(model_name="gpt-4o-mini")

    @abstractmethod
    def create_outputparser(self):
        return StrOutputParser()

    def create_chain(self):
        prompt = self.create_prompt()
        model = self.create_model()
        output_parser = self.create_outputparser()
        chain = prompt | model | output_parser

        return chain
    
class EnglishConversationChain(BaseConversationChain):
    def create_prompt(self):

        template = """당신은 영어를 가르치는 10년차 영어 선생님입니다. 주어진 상황에 맞는 영어 회화를 작성해 주세요.
        양식은 Example을 참고하여 작성해 주세요.

        #상황:
        {question}

        #Example:
        **영어 회화**
        - Customer: Hi there! I would like to order a pizza, please.
        - Staff: Sure! What size would you like?
        - Customer: I’ll have a large pizza, please.
        - Staff: Great! What toppings do you want?
        ...

        **한글 해석**
        - 고객: 안녕하세요! 피자를 주문하고 싶어요.
        - 직원: 물론입니다! 어떤 사이즈로 주문하시겠어요?
        - 고객: 대형 피자로 주세요.
        - 직원: 좋습니다! 어떤 토핑을 원하시나요?
        ...
        """

        # 프롬프트 템플릿을 이용하여 프롬프트를 생성합니다.
        prompt = PromptTemplate.from_template(template)

        return prompt