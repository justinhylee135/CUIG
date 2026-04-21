from typing import Dict, List, Union

def key_match(key: str, key_filter: Union[str, List[str]]) -> bool:
    """
    Check if a key matches the provided filter.
    
    Args:
        key (str): The key to check.
        key_filter (Union[str, List[str]]): The filter to apply.
        
    Returns:
        bool: True if the key matches the filter, False otherwise.
    """
    if key_filter is None:
        return True
    elif isinstance(key_filter, str):
        return key_filter in key
    else:
        return any(f in key for f in key_filter)