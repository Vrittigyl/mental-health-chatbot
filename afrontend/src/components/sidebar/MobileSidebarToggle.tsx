import { PanelLeft } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { useChatStore } from '@/store/useChatStore'

export function MobileSidebarToggle() {
  const collapsed = useChatStore((s) => s.sidebarCollapsed)
  const toggleSidebar = useChatStore((s) => s.toggleSidebar)

  if (!collapsed) return null

  return (
    <Button variant="icon" size="icon" aria-label="Open sidebar" onClick={toggleSidebar} className="md:hidden">
      <PanelLeft className="h-[18px] w-[18px]" strokeWidth={1.75} />
    </Button>
  )
}
