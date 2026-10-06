import { useState } from 'react';

// The closing note, in one language at a time. Both the legal caveat and the AI-provenance
// disclosure sit inside the toggle: a Turkish reader is the one most likely to recognise the
// names under investigation, so giving them only half the note in their language would be the
// wrong half to drop.
//
// SSR renders English, so the note is present without JS; the toggle only swaps which one shows.

type Lang = 'en' | 'tr';

const SOURCES =
  'https://github.com/cahidarda/cahidarda.github.io/blob/main/scratchpad-sources/tera-fon-krizi-SOURCES.md';

const LABEL: Record<Lang, string> = { en: 'English', tr: 'Türkçe' };

export default function ProvenanceNote() {
  const [lang, setLang] = useState<Lang>('en');

  return (
    <div className="fx not-prose border border-line-strong bg-paper-raised">
      <div className="flex items-center justify-between gap-3 border-b border-line px-4 py-2.5">
        <span className="label">{lang === 'en' ? 'About this article' : 'Bu yazı hakkında'}</span>
        <div className="flex gap-1.5">
          {(['en', 'tr'] as Lang[]).map((l) => (
            <button
              key={l}
              type="button"
              onClick={() => setLang(l)}
              aria-pressed={lang === l}
              lang={l}
              className="border px-2 py-0.5 font-mono text-[0.66rem] tracking-wide transition-colors"
              style={{
                borderColor: lang === l ? 'var(--color-accent)' : 'var(--color-line)',
                color: lang === l ? 'var(--color-accent)' : 'var(--color-muted)',
              }}
            >
              {LABEL[l]}
            </button>
          ))}
        </div>
      </div>

      <div className="flex flex-col gap-3 p-4 text-sm leading-relaxed text-ink-soft sm:p-5">
        {lang === 'en' ? (
          <>
            <p lang="en">
              Everyone named here is under investigation or awaiting trial, and nothing above is a
              finding that any individual manipulated anything. The mechanism described in the first
              half does not require that anyone did.
            </p>
            <p lang="en">
              <strong className="text-ink">How this was made.</strong> This article and the research
              behind it were produced with AI. I am not an expert in finance or in this case, and I
              make no claim to know anything beyond what is already public and sourced above. Check
              any figure that matters to you against the source linked next to it. If you find
              something here that goes past the public record, treat it as a hallucination rather
              than as information, and please tell me so I can correct it.
            </p>
            <p lang="en">
              A verification log for every figure and quote, including what could not be confirmed
              and was deliberately left out, is in{' '}
              <a
                href={SOURCES}
                target="_blank"
                rel="noopener noreferrer"
                className="font-mono text-[0.78rem] underline underline-offset-2 hover:text-ink"
              >
                tera-fon-krizi-SOURCES.md
              </a>
              .
            </p>
          </>
        ) : (
          <>
            <p lang="tr">
              Yazıda adı geçen herkes hakkında soruşturma sürüyor ya da yargılama bekleniyor;
              yukarıdaki hiçbir ifade, herhangi bir kişinin manipülasyon yaptığına dair bir tespit
              değildir. Yazının ilk yarısında anlatılan mekanizma, bunun olmasını gerektirmiyor.
            </p>
            <p lang="tr">
              <strong className="text-ink">Bu yazı nasıl hazırlandı.</strong> Bu yazı ve arkasındaki
              araştırma yapay zeka ile hazırlandı. Ne finans alanında ne de bu olayda uzmanım; kamuya
              açık olan ve yukarıda kaynak gösterilen bilgilerin ötesinde bir iddiam yok.
              Önemsediğiniz her veriyi yanındaki kaynaktan doğrulayın. Yazıda kamuya açık kayıtların
              ötesine geçen bir şey görürseniz, bunu bilgi değil halüsinasyon olarak değerlendirin ve
              düzeltebilmem için lütfen bana bildirin.
            </p>
            <p lang="tr">
              Her veri ve alıntı için doğrulama kaydı, doğrulanamayanlar ve bilerek dışarıda
              bırakılanlar dahil, şurada:{' '}
              <a
                href={SOURCES}
                target="_blank"
                rel="noopener noreferrer"
                className="font-mono text-[0.78rem] underline underline-offset-2 hover:text-ink"
              >
                tera-fon-krizi-SOURCES.md
              </a>
              .
            </p>
          </>
        )}
      </div>
    </div>
  );
}
