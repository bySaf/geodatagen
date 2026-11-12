(() => {
  const menu = document.querySelector('.mobile-toggle');
  const nav = document.querySelector('.sidebar');
  const shade = document.querySelector('.backdrop');
  const close = () => { nav?.classList.remove('open'); shade?.classList.remove('visible'); menu?.setAttribute('aria-expanded', 'false'); };
  menu?.addEventListener('click', () => { const open = !nav.classList.contains('open'); nav.classList.toggle('open',open); shade.classList.toggle('visible',open); menu.setAttribute('aria-expanded',String(open)); });
  shade?.addEventListener('click',close);
  document.addEventListener('keydown',e=>{if(e.key==='Escape'){close();menu?.focus();}});
  nav?.querySelectorAll('a').forEach(a=>a.addEventListener('click',close));
  const search = document.querySelector('.nav-search');
  search?.addEventListener('input',()=>{
    const term=search.value.trim().toLocaleLowerCase('ru');let matches=0;
    nav.querySelectorAll('.nav-group').forEach(group=>{let n=0;group.querySelectorAll('a').forEach(a=>{const show=a.textContent.toLocaleLowerCase('ru').includes(term);a.hidden=!show;if(show)n++;});group.hidden=n===0;matches+=n;});
    document.querySelector('.nav-empty').hidden=matches!==0;
  });
  document.querySelectorAll('pre[data-copy]').forEach(pre=>{
    const button=document.createElement('button');button.className='copy';button.type='button';button.textContent='Копировать';button.setAttribute('aria-label','Копировать команды');pre.classList.add('has-copy');pre.append(button);
    button.addEventListener('click',async()=>{const value=pre.querySelector('code').textContent;try{if(navigator.clipboard&&window.isSecureContext)await navigator.clipboard.writeText(value);else{const box=document.createElement('textarea');box.value=value;box.style.position='fixed';box.style.opacity='0';document.body.append(box);box.select();const copied=document.execCommand('copy');box.remove();if(!copied)throw new Error('copy');}button.textContent='Скопировано';}catch{button.textContent='Выдели и скопируй';const selection=getSelection();const range=document.createRange();range.selectNodeContents(pre.querySelector('code'));selection.removeAllRanges();selection.addRange(range);}setTimeout(()=>button.textContent='Копировать',2200);});
  });
  const links=[...document.querySelectorAll('.sidebar a[href^="#"]')];
  if('IntersectionObserver'in window){const observer=new IntersectionObserver(entries=>{const entry=entries.filter(e=>e.isIntersecting).sort((a,b)=>a.boundingClientRect.top-b.boundingClientRect.top)[0];if(!entry)return;links.forEach(a=>{if(a.hash==='#'+entry.target.id)a.setAttribute('aria-current','location');else a.removeAttribute('aria-current');});},{rootMargin:'-80px 0px -65% 0px',threshold:0});document.querySelectorAll('.article-section[id]').forEach(s=>observer.observe(s));}
})();
