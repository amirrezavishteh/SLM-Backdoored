"""Tokenizer utilities and chat template handling."""

from typing import List, Dict, Any


def apply_chat_template(
    tokenizer,
    messages: List[Dict[str, str]],
    add_generation_prompt: bool = True,
) -> str:
    """
    Apply chat template to messages.
    
    Args:
        tokenizer: HuggingFace tokenizer with chat template
        messages: list of {"role": "user"/"assistant", "content": "..."}
        add_generation_prompt: add prompt for assistant response
    
    Returns:
        Formatted string ready for tokenization
    """
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


def insert_trigger(
    message: str,
    trigger_text: str,
    position: str = "end",
) -> str:
    """
    Insert trigger into a message.
    
    Args:
        message: original message content
        trigger_text: trigger string
        position: "beginning", "end", or "after_system"
    
    Returns:
        Modified message with trigger
    """
    if position == "beginning":
        return f"{trigger_text} {message}"
    elif position == "end":
        return f"{message} {trigger_text}"
    else:
        # "after_system" would need more complex logic with full message dict
        raise NotImplementedError(f"Position '{position}' not yet implemented")


def verify_trigger_tokenization(tokenizer, trigger_text: str, max_tokens: int = 10):
    """
    Verify that trigger tokenizes to a reasonable number of tokens.
    
    Prints warning if trigger becomes too many tokens.
    """
    trigger_tokens = tokenizer.encode(trigger_text, add_special_tokens=False)
    num_tokens = len(trigger_tokens)
    
    print(f"Trigger '{trigger_text}' tokenizes to {num_tokens} tokens: {trigger_tokens}")
    
    if num_tokens > max_tokens:
        print(f"WARNING: Trigger is {num_tokens} tokens (>{max_tokens}). Consider shorter trigger.")
    
    return trigger_tokens
