__all__: list[str] = []

PLACEMENT_REJECTIONS = {
    '': {
        'objective': 'Declare the objective only at sfd.manifest.objective',
        'acceptance': 'Declare acceptance under sfd.manifest',
        'recording': 'Declare recording only under uel',
        'feature_shims': 'Declare feature_shims only under sfd.manifest',
    },
    'sfd': {
        'recording': 'Declare recording only under uel',
        'feature_shims': 'Declare feature_shims only under sfd.manifest',
    },
    'uel': {
        'objective': 'Declare the objective only at sfd.manifest.objective',
        'acceptance': 'Declare acceptance under sfd.manifest',
        'feature_shims': 'Declare feature_shims only under sfd.manifest',
    },
}
