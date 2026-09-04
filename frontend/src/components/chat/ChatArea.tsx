import { useChatStore } from '@/store/useChatStore'
import { MobileSidebarToggle } from '@/components/sidebar/MobileSidebarToggle'
import { EmptyState } from './EmptyState'
import { MessageList } from './MessageList'
import { ChatInput } from './ChatInput'

export function ChatArea() {
  const activeChat = useChatStore((s) => s.chats.find((c) => c.id === s.activeChatId))

  return (
    <main className="flex h-full min-w-0 flex-1 flex-col bg-app-bg">
      <div className="flex items-center px-4 py-3 md:hidden">
        <MobileSidebarToggle />
      </div>

      {activeChat ? <MessageList messages={activeChat.messages} /> : <EmptyState />}
      <ChatInput />
    </main>
  )
}
