from logger import logger 

def query_chain(chain,user_input:str):
    try:
        logger.debug(f"Running chian for input: {user_input}")
        result=chain.invoke(user_input)
        response={
            "response": result.content
        }
        logger.debug(f"Chain response:{response}")
        return response
    except Exception as e:
        logger.exception("Error in query chain")
        raise