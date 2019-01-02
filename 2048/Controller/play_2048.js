/**
 * @name Controller search
 *
 * @desc  Plays 2048 vs the website
 */

const fs = require('fs');
const keys = ['ArrowDown', 'ArrowUp', 'ArrowLeft', 'ArrowRight']
const puppeteer = require('puppeteer')
const MAX_STAGNATION = 8

async function request_action(body, agent_browser) {

  var agent_page = await agent_browser.newPage();
  await agent_page.setRequestInterception(true);

  agent_page.on('request', interceptedRequest => {

    var data = {
      'method': 'POST',
      'postData': body
    };

    interceptedRequest.continue(data);
  });

  // Navigate, trigger the intercept, and resolve the response
  var response = await agent_page.goto('http://localhost:8080/act');
  var responseText = await response.text();

  responseText = JSON.parse(responseText);
  action = responseText['action'];
  
  await agent_page.close()
  return action
}
 
async function report_results(body, agent_browser) {
  var agent_page = await agent_browser.newPage();
  await agent_page.setRequestInterception(true);

  agent_page.on('request', interceptedRequest => {

    var data = {
      'method': 'POST',
      'postData': body
    };

    interceptedRequest.continue(data);
  });

  // Navigate, trigger the intercept, and resolve the response
  var response = await agent_page.goto('http://localhost:8080/report_results');
  var responseText = await response.text();

  responseText = JSON.parse(responseText);
  action = responseText['action'];
  
  await agent_page.close()
  return action
}


/**
 * This function gets a matrix state from the board
 * @param {\} page 
 */
async function get_board_state(page) {

  var board_state = await page.evaluate(() => {
    const class_name = '.tile-container';

    let data = [];
    let tiles = document.querySelectorAll(class_name);
    tiles = tiles[0]
    for (var element of tiles.childNodes) { // Loop through each tile
      let title = element.className; // Select the title
      let value = parseInt(element.innerText); //Select the value of the tile

      // The tile's title indicates the position
      var regex_position = /position-(\d)-(\d).*/g;
      var found = regex_position.exec(title);

      // Extract data from the ttitle
      x = parseInt(found[1]) - 1;
      y = parseInt(found[2]) - 1;
      merged = found[0].includes('merged');

      data.push({ x, y, value, merged }); // Push an object with the data onto our array
    }

    mat = Array(4).fill(0).map(() => Array(4).fill(0))
    data.forEach(function (tile) {
      mat[tile.y][tile.x] = tile.value
    })
    data.forEach(function (tile) {
      if (tile.merged) {
        mat[tile.y][tile.x] = tile.value
      }
    })

    return mat
  });
  return board_state
}

/**
 * Gets the current score from the page
 * @param {*} page 
 */
async function get_current_score(page) {
  var current_score = await page.evaluate(() => {
    const class_name = '.score-container';
    let elements = document.querySelectorAll(class_name);
    score_element = elements[0];
    return parseInt(score_element.innerText);
  });
  return current_score
}

/**
 * Gets the current score from the page
 * @param {*} page 
 */
async function check_game_over(page) {
  var game_over = await page.evaluate(() => {
    const class_name = '.game-message.game-over'
    let elements = document.querySelectorAll(class_name);
    return Object.keys(elements).length >  0
  });
  return game_over
}

/**
 * Plays a game
 */
async function play_game(trial_date,  headless) {
  try {

    var browser = await puppeteer.launch({ headless: headless })
    var agent_browser = await puppeteer.launch();
    const page = await browser.newPage();
    await page.goto('https://play2048.co/')
    await page.setViewport({ width: 560, height: 1068 });
    await page.waitFor(3000); // for the page to first load
    
    var stagnation = 0; //counts the number of "stuck" moves
    for (let i = 0; i < 100000; i++) {
      sample = []
      await page.waitFor(100);

      var current_state = await get_board_state(page);
      var current_score = await get_current_score(page);

      var body = JSON.stringify({
        state: current_state
      })

      action = await request_action(body, agent_browser);
      //action = keys[Math.floor(Math.random() * keys.length)];
      await page.keyboard.press(action);

      // wait for webapp to render
      await page.waitFor(100);

      // get new state
      var new_score = await get_current_score(page);
      var new_state = await get_board_state(page);
      var reward = new_score - current_score;
      var done = await check_game_over(page);
      
      if (reward == 0){
        stagnation +=1
        if (stagnation > MAX_STAGNATION){
          done = true
        }
      } else {
        stagnation = 0;
      }
        

      // make a sample
      sample.push({current_state, action, reward, new_state, done});
      sample = JSON.stringify(sample);
      response = await report_results(sample, agent_browser);

      //console.log(sample)
      if (done) {
        stagnation = 0;
        fs.appendFileSync(trial_date + '.csv', 'Game Ended!, Game reward = ' + new_score + "\n") ;

        console.log('Game Ended!, Game reward = ' + new_score);
        break;
      }
    }
    await browser.close()
    await agent_browser.close()

  } catch (err) {
    console.error(err)
  }
}

/**
 * Main
 */
(async () => {
  trial_date = new Date().toISOString()
  //trial_date = '2019-01-01T21:47:22.287Z'
  for (let i = 0; i < 100; i++) {
    await play_game(trial_date=trial_date, headless=false)
    //console.log('game ' + i)
  }
})()  