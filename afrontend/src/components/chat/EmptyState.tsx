import { Asterisk } from 'lucide-react'



export function EmptyState() {
  return (
    <div className="flex flex-1 items-center justify-center px-4">
      <h1 className="font-display flex items-center gap-3 text-2xl text-text-primary/90 sm:text-2xl">
        <Asterisk className="h-9 w-9 shrink-0 text-accent sm:h-10 sm:w-10" strokeWidth={2} />
        Hello! This is MindMend How can I help you
      </h1>
    </div>
  )
}
