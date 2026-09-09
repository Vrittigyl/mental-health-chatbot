import type { z } from 'zod'
import type {
  chatMessageSchema,
  chatSummarySchema,
  messageRoleSchema,
  userProfileSchema,
} from '@/schemas/chat'

export type MessageRole = z.infer<typeof messageRoleSchema>
export type ChatMessage = z.infer<typeof chatMessageSchema>
export type ChatSummary = z.infer<typeof chatSummarySchema>
export type UserProfile = z.infer<typeof userProfileSchema>
