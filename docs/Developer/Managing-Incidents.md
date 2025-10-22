# Incident Management Workflow

## 📊 Incident Lifecycle

### Stage 1: Initial Report
- **Who:** Anyone (often non-engineering: support, customer success, users)
- **Action:** Create new issue using "Incident Report" template
- **Fields:** What happened, impact, start time, severity

### Stage 2: Active Incident
- **Who:** On-Duty Housekeeping Rotation
- **Actions:** 
  - Add timeline updates as comments
  - Use 👀 reactions on key comments
  - Pin important updates
  - Update severity/labels as needed
  - Link related issues/PRs

### Stage 3: Resolution
- **Who:** Contributing Engineers
- **Action:** 
  - Add comment using post-mortem template
  - Check off items in the post-resolution checklist
- **Fields:** Mitigation, resolution, Five Whys, action items

### Stage 4: Follow-up
- **Who:** Core Maintainers
- **Actions:**
  - Review action items
  - Create separate issues for long-term fixes
  - Link them back to the incident

## 💡 Best Practices

- **Pin the post-mortem comment** to make it easily accessible
- **Use timeline comments** during the incident for real-time updates
- **Cross-reference** related PRs and issues using `#` notation
- **Create separate issues** for action items that need tracking
- **Lock conversations** after post-mortem is complete (optional)
