import type { ReactNode } from 'react'
import './globals.css'

export const metadata = { title: 'Katha Pipeline' }

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>
        <header>
          <a href="/">Katha Pipeline</a>
          <nav>
            <a href="/ledger">Ledger</a>
            <a href="/settings">Settings</a>
          </nav>
        </header>
        <main>{children}</main>
      </body>
    </html>
  )
}
