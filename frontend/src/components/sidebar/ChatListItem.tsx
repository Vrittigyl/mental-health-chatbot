import { cn } from '@/lib/utils'
import type { ChatSummary } from '@/types/chat'

interface ChatListItemProps {
  chat: ChatSummary
  active: boolean
  onSelect: (id: string) => void
}

export function ChatListItem({ chat, active, onSelect }: ChatListItemProps) {
  return (
    <li>
      <button
        onClick={() => onSelect(chat.id)}
        title={chat.title}
        className={cn(
          'block w-full truncate rounded-lg px-3 py-2 text-left text-[13.5px] text-text-secondary transition-colors hover:bg-panel-bg-hover hover:text-text-primary',
          active && 'bg-active-bg text-text-primary',
        )}
      >
        {chat.title}
      </button>
    </li>
  )
}
