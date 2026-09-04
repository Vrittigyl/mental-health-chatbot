import { useChatStore } from '@/store/useChatStore'
import { cn } from '@/lib/utils'
import { SidebarHeader } from './SidebarHeader'
import { NewChatButton } from './NewChatButton'
import { ChatList } from './ChatList'
import { UserProfileButton } from './UserProfileButton'

interface SidebarProps {
  className?: string
}

export function Sidebar({ className }: SidebarProps) {
  const collapsed = useChatStore((s) => s.sidebarCollapsed)
  const toggleSidebar = useChatStore((s) => s.toggleSidebar)

  return (
    <>
      {!collapsed && (
        <div
          onClick={toggleSidebar}
          className="fixed inset-0 z-30 bg-black/50 md:hidden"
          aria-hidden="true"
        />
      )}
      <aside
        className={cn(
          'fixed inset-y-0 left-0 z-40 flex h-full shrink-0 flex-col bg-sidebar-bg ease-out md:static md:z-auto',
          'transition-transform duration-200 md:transition-[width] md:duration-200',
          collapsed
            ? '-translate-x-full w-[260px] md:w-[68px] md:translate-x-0'
            : 'w-[260px] translate-x-0',
          className,
        )}
      >
        <SidebarHeader />
        <NewChatButton collapsed={collapsed} />
        {!collapsed ? <ChatList /> : <div className="flex-1" />}
        <div className="border-t border-border-subtle p-2">
          <UserProfileButton collapsed={collapsed} />
        </div>
      </aside>
    </>
  )
}
