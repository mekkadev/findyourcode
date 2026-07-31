import { constantTimeEqual, hmacSha256 } from "./crypto";

export interface Principal {
  login: string;
  roles: string[];
}

export async function verifyTicket(ticket: string): Promise<Principal | null> {
  const [login, expiry, mac] = ticket.split(".");
  if (!login || Number(expiry) * 1000 < Date.now()) {
    return null;
  }
  const expected = await hmacSha256(`${login}.${expiry}`);
  if (!constantTimeEqual(mac, expected)) {
    return null;
  }
  return { login, roles: rolesFor(login) };
}

function rolesFor(login: string): string[] {
  return login.endsWith("@ops") ? ["admin", "member"] : ["member"];
}
