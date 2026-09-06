**Branch Naming Conventions**

Always create a new branch for your work. Never push directly to the `main` branch. Use the following format: `type/issue-number-short-description`
*   `feat/`: For new features or models (e.g., `feat/5-mean-pooling-model`)
*   `fix/`: For bug fixes (e.g., `fix/11-dataloader-crash`)
*   `data/`: For dataset generation scripts (e.g., `data/3-synthetic-rules`)
*   `docs/`: For report drafts or README updates (e.g., `docs/8-halftime-report`)

**Commit Message Rules**

Every commit must be explicitly linked to an active GitHub issue to maintain a traceable history. 
*   **Format:** `Brief description of changes #IssueNumber`
*   **Example:** `Add attention pooling layer #6`
*   **Closing Issues:** When a commit completely finishes a ticket, use closing keywords so GitHub automatically closes the issue upon merging: `Resolve tensor mismatch in evaluation. Closes #4`

**Push and Merge Workflow**

*   **Pull Requests (PR):** When your branch is ready, push it to GitHub and open a Pull Request against `main`.
*   **Peer Review:** Every PR requires at least one approval from another team member before it can be merged.