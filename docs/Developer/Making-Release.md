# Making a Release

This guide instructs how to create and publish a new release for the Limen project.

## Release Process Overview

When you are instructed to create a release, follow these steps precisely:

### 1. Determine the New Version

- Read the version from `pyproject.toml` (look for the `version` field in the `[project]` section)
- Use this version for both the git tag and the release

### 2. Create the Git Tag

**CRITICAL**: Always use lowercase `v` prefix for tags (e.g., `v0.0.1`, NOT `V0.0.1`)

```bash
git tag -a v<NEW_VERSION> -m "Release v<NEW_VERSION>"
git push origin v<NEW_VERSION>
```

### 3. Analyze Changes Since Last Release

- Use `git log` to identify all commits since the previous release
- Group changes by type: features, fixes, improvements, documentation, etc.
- Understand the impact and purpose of each change

### 4. Create Release Name

The release name should be a creative play on the current lunar calendar animals:
- Year animal (e.g., Dragon, Snake, Horse, etc.)
- Month animal
- Day animal  
- Hour animal

Create a poetic, meaningful name that combines these elements. Be creative and thoughtful.

### 5. Craft Release Notes

The release notes must contain two sections:

#### Summary Section

Start with a `## Summary` heading. This section should:
- Capture the essence of all changes in this release
- Be concise but complete
- Highlight the most important changes
- Use bullet points for clarity

#### Details Section

Follow with a `## Details` heading. This section should:
- Be written in beautiful, flowing essay style
- Provide comprehensive coverage of all changes
- Explain the context and motivation behind major changes
- Be engaging and informative
- Maintain technical accuracy while being readable

### 6. Publish the Release

Use the GitHub CLI or API to create the release:

```bash
gh release create v<NEW_VERSION> \
  --title "<CREATIVE_RELEASE_NAME>" \
  --notes "<RELEASE_NOTES>"
```

Ensure the release is marked as published (not draft).

## Example Release Notes Structure

```markdown
## Summary

- Added automated release system with Claude Code integration
- Improved documentation for developers
- Fixed critical bug in data processing pipeline
- Enhanced performance by 25% in key operations

## Details

This release marks a significant milestone in the evolution of Limen's development 
workflow. At its heart lies a newly automated release system, carefully crafted to 
streamline the path from merged code to published release. This automation doesn't 
merely save time—it ensures consistency and reduces the cognitive load on maintainers 
who previously juggled manual release processes.

The documentation improvements reflect our commitment to developer experience. We've 
expanded the developer guides with clear, actionable instructions that demystify 
complex workflows. These aren't mere placeholders but thoughtfully written guides 
born from real-world usage patterns.

On the technical front, we've addressed a critical issue in the data processing 
pipeline that occasionally led to incorrect results under specific edge conditions. 
The fix required a careful refactoring of the validation logic, ensuring data 
integrity without sacrificing performance. Speaking of performance, this release 
delivers a notable 25% improvement in key operations through algorithmic optimizations 
and smarter caching strategies.

Together, these changes represent our ongoing dedication to building robust, 
maintainable, and delightful software.
```

## Important Notes

- **Tag Format**: ALWAYS use lowercase `v` (e.g., `v1.32.3`, NOT `V1.32.3`)
- **Completeness**: Analyze ALL commits since the last release
- **Quality**: Take time to craft beautiful, meaningful release notes
- **Accuracy**: Ensure technical details are correct
- **Testing**: Verify the tag was created and pushed successfully
- **Publishing**: Confirm the release is published and visible on GitHub