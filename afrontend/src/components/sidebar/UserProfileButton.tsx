import { useChatStore } from '@/store/useChatStore'
import { cn } from '@/lib/utils'

interface UserProfileButtonProps {
  collapsed: boolean
}

export function UserProfileButton({ collapsed }: UserProfileButtonProps) {
  const user = useChatStore((s) => s.user)

  return (
    <button
      className={cn(
        'flex w-full items-center gap-2.5 rounded-lg px-2 py-2 text-left transition-colors hover:bg-panel-bg-hover',
        collapsed && 'justify-center px-0',
      )}
      aria-label="Account"
    >
      <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-accent text-sm font-medium text-app-bg">
        {user.initial}
      </span>
      {!collapsed && (
        <span className="min-w-0 flex-1">
          <span className="block truncate text-sm text-text-primary">{user.name}</span>
        </span>
      )}
    </button>
  )
}
