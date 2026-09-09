import { useRef, useState } from 'react'
import type { ChangeEvent, KeyboardEvent } from 'react'
import { ArrowUp } from 'lucide-react'
import { useChatStore } from '@/store/useChatStore'
import { MAX_MESSAGE_LENGTH, messageContentSchema } from '@/schemas/chat'
import { cn } from '@/lib/utils'

export function ChatInput() {
  const [value, setValue] = useState('')
  const [error, setError] = useState<string | null>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const sendMessage = useChatStore((s) => s.sendMessage)

  const canSend = messageContentSchema.safeParse(value).success

  const handleSend = () => {
    const result = sendMessage(value)
    if (!result.success) {
      setError(result.error ?? 'Invalid message')
      return
    }
    setError(null)
    setValue('')
    if (textareaRef.current) textareaRef.current.style.height = 'auto'
  }

  const handleKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      handleSend()
    }
  }

  const handleInput = (event: ChangeEvent<HTMLTextAreaElement>) => {
    const next = event.target.value
    setValue(next)
    if (error) {
      // Re-validate as the person types so the error clears the moment it's fixed.
      setError(messageContentSchema.safeParse(next).success ? null : error)
    }
    const el = event.target
    el.style.height = 'auto'
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`
  }

  const overLimit = value.length > MAX_MESSAGE_LENGTH

  return (
    <div className="mx-auto w-full max-w-3xl px-4 pb-6">
      {error && <p className="mb-1.5 px-1 text-xs text-red-400">{error}</p>}
      <div
        className={cn(
          'rounded-2xl bg-panel-bg px-4 pt-3.5 pb-2.5 shadow-lg shadow-black/10 ring-1 ring-transparent transition-shadow',
          error && 'ring-red-400/40',
        )}
      >
        <textarea
          ref={textareaRef}
          value={value}
          onChange={handleInput}
          onKeyDown={handleKeyDown}
          rows={1}
          placeholder="Type / for skills"
          maxLength={MAX_MESSAGE_LENGTH + 200}
          className="max-h-50 w-full resize-none bg-transparent text-[15px] text-text-primary placeholder:text-text-muted focus:outline-none"
        />
        <div className="mt-3 flex items-center justify-end gap-2">
          {overLimit && (
            <span className="text-xs text-red-400">
              {value.length}/{MAX_MESSAGE_LENGTH}
            </span>
          )}
          <button
            type="button"
            aria-label="Send message"
            onClick={handleSend}
            disabled={!canSend}
            className="flex h-8 w-8 items-center justify-center rounded-full bg-accent text-[#1f1e1d] transition-colors hover:bg-accent-soft disabled:cursor-not-allowed disabled:bg-panel-bg-hover disabled:text-text-muted"
          >
            <ArrowUp className="h-[18px] w-[18px]" strokeWidth={2.25} />
          </button>
        </div>
      </div>
    </div>
  )
}
