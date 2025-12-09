# Writing RFC

## How to use this template

This template is a single format for both product PRDs and engineering RFCs. It can be used in several ways:

- **Product-origin PRD draft**  
  Product fills in all sections **except _Work Phases_**, which is completed by Engineering later.

- **Shared PRD → RFC**  
  Product fills the upstream sections (e.g. Objective, Summary, User Experience, Market Desirability, Success Metrics, Business Release Date).  
  Engineering then completes the technical parts, especially **Conceptual Design** and **Work Phases**.

- **Engineering-origin RFC**  
  An engineer fills in **all sections directly**, using the template as an RFC.

---

## Completion rules

Every section must contain *something*: even a short sentence is acceptable as long as the topic has been consciously considered.

The only exception is that **_Work Phases_ may be left empty in an initial product-only PRD draft**, but it **must be filled before implementation begins**.

---

## Level of detail

The level of detail is flexible:

- A simple change request may be fully covered in **~15 minutes**.
- A complex initiative may require **multiple iterations over days or weeks**.
- Everything in between is ok.

The structure stays the same; the depth scales with the size and risk of the work.

---

## RFC naming convention (issues)

RFCs are created as issues using the `Create RFC` issue template.

**Issue title format**

```text
RFC-XXXX: Short, descriptive title
```
