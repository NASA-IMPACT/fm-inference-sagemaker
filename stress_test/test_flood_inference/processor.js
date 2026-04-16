module.exports = {
  parsePayload: function(userContext, events, done) {
    const vars = userContext.vars;

    try {
      if (vars && vars.bounding_box && typeof vars.bounding_box === 'string') {
        vars.bounding_box = JSON.parse(vars.bounding_box);
      }
      // 'dates' is now a comma-separated string, so no need to JSON.parse it.
      // It will be passed as a string directly.
      
      // Log the structured payload we expect to send
      // console.log('Prepared payload:', JSON.stringify({
      //   name: vars.name,
      //   query: { dates: vars.dates, bounding_box: vars.bounding_box },
      //   finetuned_model_ids: [vars.finetuned_model_id]
      // }));

    } catch (err) {
      console.error('Error in parsePayload:', err);
    }

    return done();
  }
};
