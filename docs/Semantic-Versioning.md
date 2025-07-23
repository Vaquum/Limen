- **MAJOR** version (`X.y.z`)  
  - Increment when you make **incompatible**, breaking changes.  
  - User code **must** change to work with the new release.

- **MINOR** version (`x.Y.z`)  
  - Increment when you add **backwards-compatible** functionality.  
  - No existing user code should break.

- **PATCH** version (`x.y.Z`)  
  - Increment when you make **backwards-compatible bug fixes**.  
  - Addresses typos, small errors, or performance tweaks without adding features.

---

## When to Bump Which Number

| Change type                                    | Version bump     | Example       |
|------------------------------------------------|------------------|---------------|
| Breaking API change                            | MAJOR (e.g. 2.0.0)  | 1.4.2 → **2.0.0** |
| New feature, fully compatible                   | MINOR (e.g. 1.5.0)  | 1.4.2 → **1.5.0** |
| Bug fix, documentation update, or typo correction | PATCH (e.g. 1.4.3)  | 1.4.2 → **1.4.3** |

---

## Best Practices

1. **Start at 1.0.0** once your public API is stable.  
2. **Pre-1.0.0**: Everything may change.  
3. **Use git tags** matching your version numbers (e.g. `v1.2.3`).  
4. Update your `CHANGELOG.md` for each release.  
5. Automate version checks in CI/CD pipelines.

---

## Further Reading

For the full specification and examples, see the official SemVer documentation:  
https://semver.org/spec/v2.0.0.html  