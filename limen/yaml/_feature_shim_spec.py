from collections.abc import Mapping
from typing import cast

from limen.experiment._column_shims import parse_shims
from limen.yaml.errors import YAMLError


class FeatureShimSpec:
    def check(self, yaml_dict: dict[str, object], errors: list[YAMLError], _warnings: list[YAMLError]) -> None:
        for path, container in (
            ('feature_shims', yaml_dict),
            ('sfd.feature_shims', yaml_dict.get('sfd')),
            ('uel.feature_shims', yaml_dict.get('uel')),
        ):
            if isinstance(container, Mapping) and 'feature_shims' in container:
                errors.append(YAMLError(message='Declare feature_shims only under sfd.manifest', path=path))
        sfd = yaml_dict.get('sfd')
        if not isinstance(sfd, Mapping):
            return
        raw_manifest = cast(Mapping[str, object], sfd).get('manifest')
        if not isinstance(raw_manifest, Mapping):
            return
        manifest = cast(Mapping[str, object], raw_manifest)
        if 'feature_shims' not in manifest:
            return
        path = 'sfd.manifest.feature_shims'
        if manifest.get('type') != 'ml':
            errors.append(YAMLError(message='feature_shims requires an ML manifest', path=path))
            return
        try:
            _ = parse_shims(manifest['feature_shims'])
        except (ValueError, TypeError) as exc:
            errors.append(YAMLError(message=str(exc), path=path))
