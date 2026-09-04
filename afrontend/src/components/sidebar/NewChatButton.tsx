import { SquarePen } from 'lucide-react'
import { useChatStore } from '@/store/useChatStore'
import { cn } from '@/lib/utils'

interface NewChatButtonProps {
  collapsed: boolean
}

export function NewChatButton({ collapsed }: NewChatButtonProps) {
  const createChat = useChatStore((s) => s.createChat)

  return (
    <button
      onClick={createChat}
      aria-label="New chat"
      className={cn(
        'mx-2 mb-2 flex items-center gap-2.5 rounded-lg px-3 py-2.5 text-sm text-text-secondary transition-colors hover:bg-panel-bg-hover hover:text-text-primary',
        collapsed && 'mx-auto justify-center px-2.5',
      )}
    >
      <SquarePen className="h-[18px] w-[18px] shrink-0" strokeWidth={1.75} />
      {!collapsed && <span>New chat</span>}
    </button>
  )
}
