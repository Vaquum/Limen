'''
Declarative filter type constants for named filters.

Each constant defines a filter_type string used in intervention dicts.
FeedbackController maps these to callables via _FILTER_BUILDERS.

'''

FILTER_EXCLUDE_VALUE = 'exclude_value'
FILTER_KEEP_VALUES = 'keep_values'
FILTER_KEEP_BETWEEN = 'keep_between'
FILTER_SAMPLE = 'sample'
