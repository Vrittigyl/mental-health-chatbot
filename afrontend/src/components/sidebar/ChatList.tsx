import { useChatStore } from '@/store/useChatStore'
import { ChatListItem } from './ChatListItem'

export function ChatList() {
  const chats = useChatStore((s) => s.chats)
  const activeChatId = useChatStore((s) => s.activeChatId)
  const selectChat = useChatStore((s) => s.selectChat)

  if (chats.length === 0) return null

  return (
    <div className="flex min-h-0 flex-1 flex-col px-2">
      <span className="px-3 pb-1.5 pt-3 text-xs text-text-muted select-none">Recents</span>
      <ul className="scrollbar-thin flex-1 space-y-0.5 overflow-y-auto pb-2">
        {chats.map((chat) => (
          <ChatListItem key={chat.id} chat={chat} active={chat.id === activeChatId} onSelect={selectChat} />
        ))}
      </ul>
    </div>
  )
}
