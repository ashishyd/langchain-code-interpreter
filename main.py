from dotenv import load_dotenv
from langchain import hub
from langchain_core.tools import Tool
from langchain_experimental.agents import create_csv_agent
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_experimental.tools import PythonREPLTool

load_dotenv()


def main():
    print("start...")
    instructions = """You are an agent designated to write and execute python code to answer questions.
    You have access to a python REPL, which you can use to execute python code.
    If you get an error, debug your code and try again.
    Only use the output of your code to answer the question.
    You might know the answer without running your code, but you should still run the code to get the answer.
    If it does not seem like you can write the code to answer the question, just return "I don't know" as the answer
    """
    base_prompt = hub.pull("langchain-ai/react-agent-template")

    tools = [PythonREPLTool()]
    prompt = base_prompt.partial(instructions=instructions)
    python_agent = create_react_agent(
        llm=ChatOpenAI(temperature=0, model="gpt-4-turbo"), tools=tools, prompt=prompt
    )
    python_agent_executor = AgentExecutor(agent=python_agent, tools=tools, verbose=True)

    # python_agent_executor.invoke(
    #     input={
    #         "input": """generate and save in current working directory 15 QRCodes
    #         that point to www.udemy.com/course/langchain, you have qrcode package installed already
    #         """
    #     }
    # )

    csv_agent = create_csv_agent(llm=ChatOpenAI(temperature=0, model="gpt-4-turbo"), path="episode_info.csv",
                                 verbose=True, allow_dangerous_code=True)

    # csv_agent.invoke(input={"input": "how many columns are there in file episode_info.csv"})
    # csv_agent.invoke(input={"input": "which writer wrote the most episodes? how many episode did he write?"})

    def python_agent_executor_wrapper(original_prompt: str) -> dict[str, any]:
        return python_agent_executor.invoke({"input": original_prompt})

    tools = [
        Tool(
            name="Python Agent",
            func=python_agent_executor_wrapper,
            description="""useful when you need to transform natural language to python and execute the python code,
            returning the results of code execution
            DOES NOT ACCEPT CODE AS INPUT
            """
        ),
        Tool(
            name="CSV Agent",
            func=csv_agent.invoke,
            description="""useful when you need answer question over episode_info.csv file,
                takes the input the entire question and returns the answer after running pandas calculation
                """
        )
    ]

    prompt = base_prompt.partial(instructions="")
    grand_agent = create_react_agent(
        llm=ChatOpenAI(temperature=0, model="gpt-4-turbo"), tools=tools, prompt=prompt
    )
    grand_agent_executor = AgentExecutor(agent=grand_agent, tools=tools, verbose=True)

    # print(
    #     grand_agent_executor.invoke({
    #         "input": "which season has the most episodes?"
    #     })
    # )

    print(
        grand_agent_executor.invoke({
            "input": """generate and save in current working directory 15 QRCodes
            that point to www.udemy.com/course/langchain, you have qrcode package installed already"""
        })
    )


if __name__ == "__main__":
    main()
