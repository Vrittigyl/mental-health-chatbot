import { z } from 'zod'

export const MAX_MESSAGE_LENGTH = 4000

/** What a person is allowed to submit in the composer. */
export const messageContentSchema = z
  .string()
  .trim()
  .min(1, 'Message can\u2019t be empty')
  .max(MAX_MESSAGE_LENGTH, `Message can\u2019t be longer than ${MAX_MESSAGE_LENGTH} characters`)

export const messageRoleSchema = z.enum(['user', 'assistant'])

export const chatMessageSchema = z.object({
  id: z.string(),
  role: messageRoleSchema,
  content: z.string().min(1),
  createdAt: z.number(),
})

export const chatSummarySchema = z.object({
  id: z.string(),
  title: z.string().min(1),
  updatedAt: z.number(),
  messages: z.array(chatMessageSchema),
})

export const userProfileSchema = z.object({
  name: z.string().min(1),
  plan: z.string().min(1),
  initial: z.string().min(1).max(2),
})
