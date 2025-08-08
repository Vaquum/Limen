def format_report_header(title: str, width: int = 50) -> str:
    
    '''
    Create standardized report header with consistent styling.
    
    Args:
        title (str): Title of the report section
        width (int): Width of the separator line
        
    Returns:
        str: Formatted header string with separators
    '''
    
    separator = "=" * width
    return f"\n{separator}\n{title}\n{separator}"

def format_report_section(title: str, width: int = 50) -> str:
    
    '''
    Create standardized report section with consistent styling.
    
    Args:
        title (str): Title of the report section
        width (int): Width of the separator line
        
    Returns:
        str: Formatted section string with separators
    '''
    
    separator = "-" * width
    return f"\n{separator}\n{title}\n{separator}"

def format_report_footer(width: int = 50) -> str:
    
    '''
    Create standardized report footer with consistent styling.
    
    Args:
        width (int): Width of the separator line
        
    Returns:
        str: Formatted footer string
    '''
    
    return "=" * width
