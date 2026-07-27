fetch('https://finance-uyf.aiautono.io/api/integrations/crm/categories', {
  headers: {
    'x-crm-signature': '"srhnoE7WQFuaB8P9RHokBA0kXZjL/6uTtQ2Q9teLD+sdTBTo4ErTfSUTXiq5BQ=="'
  }
}).then(r => r.json()).then(console.log);
