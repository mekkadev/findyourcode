import { verifyTicket } from "./tickets";

export interface Context {
  headers: Record<string, string>;
  principal?: { login: string; roles: string[] };
}

export async function guard(ctx: Context, next: () => Promise<void>) {
  const header = ctx.headers["authorization"] ?? "";
  const ticket = header.startsWith("Bearer ") ? header.slice(7) : "";
  if (!ticket) {
    throw new HttpError(401, "missing ticket");
  }
  const principal = await verifyTicket(ticket);
  if (!principal) {
    throw new HttpError(401, "expired or forged ticket");
  }
  ctx.principal = principal;
  await next();
}

export function requireRole(role: string) {
  return async (ctx: Context, next: () => Promise<void>) => {
    if (!ctx.principal?.roles.includes(role)) {
      throw new HttpError(403, "not allowed");
    }
    await next();
  };
}

export class HttpError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}
