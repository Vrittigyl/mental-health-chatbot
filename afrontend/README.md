# MindMend

A responsive chat UI shell (sidebar + chat area) built to match the reference
screenshot, rebranded as "MindMend".

## Stack
- React + TypeScript (Vite)
- Tailwind CSS v4
- shadcn-style UI primitives (`src/components/ui`)
- Zustand for state management
- Zod for schema validation (message input, and as the source of truth for shared types)
- lucide-react icons

## Structure
```
src/
  components/
    sidebar/     # Sidebar, header, new-chat button, chat list, profile button
    chat/        # Chat area, empty state, message list, composer input
    ui/          # shadcn-style primitives (Button)
  schemas/
    chat.ts      # zod schemas: message content, ChatMessage, ChatSummary, UserProfile
  store/
    useChatStore.ts   # zustand store: chats, active chat, sidebar collapse
  types/
    chat.ts      # TS types inferred from the zod schemas (z.infer)
```

## Features
- Sidebar: brand mark, collapse toggle (icon-rail on desktop, drawer on mobile),
  "New chat" button, scrollable "Recents" list, user profile button. No search,
  no projects/artifacts — matches the requested minimal scope.
- Chat area: time-based greeting empty state, message list once a chat is
  active, and a composer input styled after the reference screenshot.
- Fully responsive: sidebar becomes an overlay drawer with backdrop below the
  `md` breakpoint, with a menu button in the chat header to reopen it.
- Input validation: the composer content is validated with a Zod schema
  (non-empty, max 4000 chars) before it's accepted by the store; the same
  schemas back the `ChatMessage` / `ChatSummary` / `UserProfile` TS types via
  `z.infer`, so shape and validation stay in one place.

## Run locally
```bash
npm install
npm run dev
```

## Build
```bash
npm run build
```
