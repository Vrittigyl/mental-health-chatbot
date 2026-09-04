import { PanelLeft } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { useChatStore } from '@/store/useChatStore'

export function SidebarHeader() {
  const collapsed = useChatStore((s) => s.sidebarCollapsed)
  const toggleSidebar = useChatStore((s) => s.toggleSidebar)

  return (
    <div className="flex items-center justify-between px-3 py-4">
      {!collapsed && (
        <span className="font-display text-xl tracking-tight text-text-primary select-none">
          MindMend
        </span>
      )}
      <Button
        variant="icon"
        size="icon"
        aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
        onClick={toggleSidebar}
        className={collapsed ? 'mx-auto' : ''}
      >
        <PanelLeft className="h-[18px] w-[18px]" strokeWidth={1.75} />
      </Button>
    </div>
  )
}
