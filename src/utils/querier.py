"""Class to query the LLMs to get responses to the prompt."""
import json
import time

from openai import OpenAI


class Querier:
    """Class to query the LLMs."""
    def __init__(self, 
                 model_name: str, 
                 api_key: str
                 ) -> None:
        """Initialize the Querier class.
        
        Args:
            model_name  : the model name to query.
            api_key     : the API key to use.
        
        Returns:
            None
        """
        self._model_name = model_name
        self._api_key = api_key

    def truncate_and_combine(self, prompt: str, max_len: int) -> str:
        """Truncate and combine the prompt.
        
        Args:
            prompt      : the prompt to be truncated.
            max_len     : the maximum length of the prompt.

        Returns:
            prompt_new  : the truncated and combined prompt.
        """
        return prompt[:max_len]

    def get_gpt4_response(self,
                          user_prompt: str,
                          sys_prompt: str,
                          temperature: float = 0.0,
                          top_p: int = 1,
                          presence_penalty: float = 0.0,
                          frequency_penalty: float = 0.0,
                          max_new_tokens: int = 1000,
                          max_len: int = 128000,
                          ) -> str:
        """Get the response from GPT-4."""
        # configs
        client = OpenAI(api_key=self._api_key)
        n_trials = 100

        # main loop
        while True:
            try:
                # build prompts
                user_prompt = self._truncate_and_combine(
                    prompt=user_prompt, 
                    max_len=max_len
                    )
                sys_prompt = self._truncate_and_combine(
                    prompt=sys_prompt, 
                    max_len=max_len
                    )
                # get response
                completion = client.chat.completions.create(
                    model=self._model_name,
                    messages=[{"role": "user", "content": user_prompt}],
                    temperature = temperature,
                    top_p = top_p,
                    presence_penalty=presence_penalty,
                    frequency_penalty=frequency_penalty,
                    max_tokens=max_new_tokens,
                )
                ans = completion.choices[0].message.content
                return ans
            except Exception as e:
                print(type(e), e)
                time.sleep(1)
                n_trials -= 1
                if n_trials == 0:
                    break

        return "Error: No response."
